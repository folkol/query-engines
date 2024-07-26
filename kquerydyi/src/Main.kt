import com.univocity.parsers.csv.CsvParser
import com.univocity.parsers.csv.CsvParserSettings
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.async
import kotlinx.coroutines.runBlocking
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.*
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType
import java.io.File
import java.io.FileNotFoundException
import java.sql.SQLException
import kotlin.NoSuchElementException
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.listOf as listOf1

private object ArrowTypes {
    val DoubleType = ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    val StringType = ArrowType.Utf8()
}

private interface ColumnVector {
    fun getValue(i: Int): Any?
    fun size(): Int
}

private data class Field(val name: String, val dataType: ArrowType) {
    fun toArrow(): org.apache.arrow.vector.types.pojo.Field {
        val fieldType = org.apache.arrow.vector.types.pojo.FieldType(true, dataType, null)
        return org.apache.arrow.vector.types.pojo.Field(name, fieldType, listOf1())
    }
}

private data class Schema(val fields: List<Field>) {

    fun toArrow(): org.apache.arrow.vector.types.pojo.Schema {
        return org.apache.arrow.vector.types.pojo.Schema(fields.map { it.toArrow() })
    }

    fun select(names: List<String>): Schema {
        val f = mutableListOf<Field>()
        names.forEach { name ->
            val m = fields.filter { it.name == name }
            if (m.size == 1) {
                f.add(m[0])
            } else {
                throw IllegalArgumentException()
            }
        }
        return Schema(f)
    }
}

private class RecordBatch(val schema: Schema, val fields: List<ColumnVector>) {
    fun rowCount() = fields.first().size()
    fun field(i: Int): ColumnVector {
        return fields[i]
    }
}

private interface DataSource {
    fun schema(): Schema
    fun scan(projection: List<String>): Sequence<RecordBatch>
}

private interface LogicalPlan {
    fun schema(): Schema
    fun children(): List<LogicalPlan>
}

private interface LogicalExpr {
    fun toField(input: LogicalPlan): Field
}

private class Column(val name: String) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return input.schema().fields.find { it.name == name } ?: throw SQLException("No column named '$name'")
    }

    override fun toString(): String {
        return "#$name"
    }
}

private abstract class AggregateExpr(
    val name: String, val expr: LogicalExpr
) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(name, expr.toField(input).dataType)
    }

    override fun toString(): String {
        return "$name($expr)"
    }
}

private class Max(input: LogicalExpr) : AggregateExpr("MAX", input)

private class Scan(
    val path: String, val dataSource: DataSource, val projection: List<String>
) : LogicalPlan {
    val schema = deriveSchema()

    override fun schema(): Schema {
        return schema
    }

    private fun deriveSchema(): Schema {
        val schema = dataSource.schema()
        return if (projection.isEmpty()) {
            schema
        } else {
            schema.select(projection)
        }
    }

    override fun children(): List<LogicalPlan> {
        return listOf1()
    }

    override fun toString(): String {
        return if (projection.isEmpty()) {
            "Scan: $path; projection=None"
        } else {
            "Scan: $path; projection=$projection"
        }
    }
}

private class Projection(
    val input: LogicalPlan, val expr: List<LogicalExpr>
) : LogicalPlan {
    override fun schema(): Schema {
        return Schema(expr.map { it.toField(input) })
    }

    override fun children(): List<LogicalPlan> {
        return listOf1(input)
    }

    override fun toString(): String {
        return "Projection: ${
            expr.joinToString(",") {
                it.toString()
            }
        }"
    }
}

private class Aggregate(
    val input: LogicalPlan, val groupExpr: List<LogicalExpr>, val aggExpr: List<AggregateExpr>
) : LogicalPlan {
    override fun schema(): Schema {
        return Schema(groupExpr.map { it.toField(input) } + aggExpr.map { it.toField(input) })
    }

    override fun children(): List<LogicalPlan> {
        return listOf1(input)
    }

    override fun toString(): String {
        return "Aggregate: groupExpr=$groupExpr, aggregateExpr=$aggExpr"
    }
}

private class ReaderAsSequence(
    private val schema: Schema, private val parser: CsvParser, private val batchSize: Int
) : Sequence<RecordBatch> {
    override fun iterator(): Iterator<RecordBatch> {
        return ReaderIterator(schema, parser, batchSize)
    }
}

private class ArrowFieldVector(val field: FieldVector) : ColumnVector {

    override fun getValue(i: Int): Any? {

        if (field.isNull(i)) {
            return null
        }

        return when (field) {
            is Float8Vector -> field.get(i)
            is VarCharVector -> {
                val bytes = field.get(i)
                if (bytes == null) {
                    null
                } else {
                    String(bytes)
                }
            }

            else -> throw IllegalStateException()
        }
    }

    override fun size(): Int {
        return field.valueCount
    }
}

private class ReaderIterator(
    private val schema: Schema, private val parser: CsvParser, private val batchSize: Int
) : Iterator<RecordBatch> {

    private var next: RecordBatch? = null
    private var started: Boolean = false

    override fun hasNext(): Boolean {
        if (!started) {
            started = true

            next = nextBatch()
        }

        return next != null
    }

    override fun next(): RecordBatch {
        if (!started) {
            hasNext()
        }

        val out = next

        next = nextBatch()

        if (out == null) {
            throw NoSuchElementException(
                "Cannot read past the end of ${ReaderIterator::class.simpleName}"
            )
        }

        return out
    }

    private fun nextBatch(): RecordBatch? {
        val rows = ArrayList<com.univocity.parsers.common.record.Record>(batchSize)

        do {
            val line = parser.parseNextRecord()
            if (line != null) rows.add(line)
        } while (line != null && rows.size < batchSize)

        if (rows.isEmpty()) {
            return null
        }

        return createBatch(rows)
    }

    private fun createBatch(rows: ArrayList<com.univocity.parsers.common.record.Record>): RecordBatch {
        val toArrow = schema.toArrow()
        val root = VectorSchemaRoot.create(toArrow, RootAllocator(Long.MAX_VALUE))
        root.fieldVectors.forEach { it.setInitialCapacity(rows.size) }
        root.allocateNew()

        root.fieldVectors.withIndex().forEach { field ->
            when (val vector = field.value) {
                is VarCharVector -> rows.withIndex().forEach { row ->
                    val valueStr = row.value.getValue(field.value.name, "").trim()
                    vector.setSafe(row.index, valueStr.toByteArray())
                }

                else -> throw IllegalStateException("No support for reading CSV columns with data type $vector")
            }
            field.value.valueCount = rows.size
        }

        return RecordBatch(schema, root.fieldVectors.map { ArrowFieldVector(it) })
    }
}

private class CsvDataSource(
    private val filename: String,
    private val hasHeaders: Boolean,
    private val batchSize: Int,
    val schema: Schema?,
) : DataSource {


    private val finalSchema: Schema by lazy { schema ?: inferSchema() }

    private fun buildParser(settings: CsvParserSettings): CsvParser {
        return CsvParser(settings)
    }

    private fun defaultSettings(): CsvParserSettings {
        return CsvParserSettings().apply {
            isDelimiterDetectionEnabled = true
            isLineSeparatorDetectionEnabled = true
            skipEmptyLines = true
            isAutoClosingEnabled = true
        }
    }

    override fun schema(): Schema {
        return finalSchema
    }

    override fun scan(projection: List<String>): Sequence<RecordBatch> {

        val file = File(filename)
        if (!file.exists()) {
            throw FileNotFoundException(file.absolutePath)
        }

        val readSchema = if (projection.isNotEmpty()) {
            finalSchema.select(projection)
        } else {
            finalSchema
        }

        val settings = defaultSettings()
        if (projection.isNotEmpty()) {
            settings.selectFields(*projection.toTypedArray())
        }
        settings.isHeaderExtractionEnabled = hasHeaders
        if (!hasHeaders) {
            settings.setHeaders(*readSchema.fields.map { it.name }.toTypedArray())
        }

        val parser = buildParser(settings)
        parser.beginParsing(file.inputStream().reader())
        parser.detectedFormat

        return ReaderAsSequence(readSchema, parser, batchSize)
    }

    private fun inferSchema(): Schema {

        val file = File(filename)
        if (!file.exists()) {
            throw FileNotFoundException(file.absolutePath)
        }

        val parser = buildParser(defaultSettings())
        return file.inputStream().use {
            parser.beginParsing(it.reader())
            parser.detectedFormat

            parser.parseNext()
            val headers = parser.context.parsedHeaders().filterNotNull()

            val schema = if (hasHeaders) {
                Schema(headers.map { colName -> Field(colName, ArrowTypes.StringType) })
            } else {
                Schema(List(headers.size) { i -> Field("field_${i + 1}", ArrowTypes.StringType) })
            }

            parser.stopParsing()
            schema
        }
    }
}

private interface DataFrame {
    fun project(expr: List<LogicalExpr>): DataFrame

    //    fun filter(expr: LogicalExpr): DataFrame
    fun aggregate(groupBy: List<LogicalExpr>, aggregateExpr: List<AggregateExpr>): DataFrame
    fun schema(): Schema
    fun logicalPlan(): LogicalPlan
}

private class DataFrameImpl(private val plan: LogicalPlan) : DataFrame {
    override fun project(expr: List<LogicalExpr>): DataFrame {
        return DataFrameImpl(Projection(plan, expr))
    }

    override fun aggregate(groupBy: List<LogicalExpr>, aggregateExpr: List<AggregateExpr>): DataFrame {
        return DataFrameImpl(Aggregate(plan, groupBy, aggregateExpr))

    }

    override fun schema(): Schema {
        return plan.schema()
    }

    override fun logicalPlan(): LogicalPlan {
        return plan
    }
}

private class ExecutionContext {
    private val tables = mutableMapOf<String, DataFrame>()

    fun sql(sql: String): DataFrame {
        val tokens = SqlTokenizer(sql).tokenize()
        val ast = SqlParser(tokens).parse() as SqlSelect
        val df = createDataFrame(ast, tables)
        return DataFrameImpl(df.logicalPlan())
    }

    private fun csv(filename: String): DataFrame {
        return DataFrameImpl(Scan(filename, CsvDataSource(filename, true, 1000, null), listOf1()))
    }

    private fun register(tablename: String, df: DataFrame) {
        tables[tablename] = df
    }

    fun registerCsv(tablename: String, filename: String) {
        register(tablename, csv(filename))
    }

    fun registerDataSource(tablename: String, datasource: DataSource) {
        register(tablename, DataFrameImpl(Scan(tablename, datasource, listOf1())))
    }

    fun execute(df: DataFrame): Sequence<RecordBatch> {
        return execute(df.logicalPlan())
    }

    private fun execute(plan: LogicalPlan): Sequence<RecordBatch> {
        val optimizedPlan = ProjectionPushDownRule().optimize(plan)
        val physicalPlan = createPhysicalPlan(optimizedPlan)
        return physicalPlan.execute()
    }
}

private class CastExpr(val expr: LogicalExpr, val dataType: ArrowType) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(expr.toField(input).name, dataType)
    }

    override fun toString(): String {
        return "CAST($expr AS $dataType)"
    }
}

private class Alias(val expr: LogicalExpr, val alias: String) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(alias, expr.toField(input).dataType)
    }

    override fun toString(): String {
        return "$expr as $alias"
    }
}

private interface PhysicalPlan {
    fun schema(): Schema
    fun execute(): Sequence<RecordBatch>
    fun children(): List<PhysicalPlan>
}

private interface Expression {
    fun evaluate(input: RecordBatch): ColumnVector
}

private class ColumnExpression(val i: Int) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        return input.field(i)
    }

    override fun toString(): String {
        return "#$i"
    }
}

object FieldVectorFactory {

    fun create(arrowType: ArrowType, initialCapacity: Int): FieldVector {
        val rootAllocator = RootAllocator(Long.MAX_VALUE)
        val fieldVector: FieldVector = when (arrowType) {
            ArrowTypes.DoubleType -> Float8Vector("v", rootAllocator)
            ArrowTypes.StringType -> VarCharVector("v", rootAllocator)
            else -> throw IllegalStateException()
        }
        if (initialCapacity != 0) {
            fieldVector.setInitialCapacity(initialCapacity)
        }
        fieldVector.allocateNew()
        return fieldVector
    }
}

private class ArrowVectorBuilder(private val fieldVector: FieldVector) {

    fun set(i: Int, value: Any?) {
        when (fieldVector) {
            is VarCharVector -> {
                if (value == null) {
                    fieldVector.setNull(i)
                } else {
                    fieldVector.set(i, value.toString().toByteArray())
                }
            }

            is Float8Vector -> {
                if (value == null) {
                    fieldVector.setNull(i)
                } else if (value is Number) {
                    fieldVector.set(i, value.toDouble())
                } else {
                    throw IllegalStateException()
                }
            }

            else -> throw IllegalStateException(fieldVector.javaClass.name)
        }
    }

    fun setValueCount(n: Int) {
        fieldVector.valueCount = n
    }

    fun build(): ColumnVector {
        return ArrowFieldVector(fieldVector)
    }
}

private interface AggregateExpression {
    fun inputExpression(): Expression
    fun createAccumulator(): Accumulator
}

private interface Accumulator {
    fun accumulate(value: Any?)
    fun finalValue(): Any?
}

private class MaxExpression(private val expr: Expression) : AggregateExpression {
    override fun inputExpression(): Expression {
        return expr
    }

    override fun createAccumulator(): Accumulator {
        return MaxAccumulator()
    }

    override fun toString(): String {
        return "MAX($expr)"
    }
}

private class MaxAccumulator : Accumulator {
    var value: Any? = null
    override fun accumulate(value: Any?) {
        if (value != null) {
            if (this.value == null) {
                this.value = value
            } else {
                val isMax = when (value) {
                    is Byte -> value > this.value as Byte
                    is Double -> value > this.value as Double
                    else -> throw UnsupportedOperationException(
                        "MAX is not implemented for data type ${value.javaClass.name}"
                    )
                }
                if (isMax) {
                    this.value = value
                }
            }
        }
    }

    override fun finalValue(): Any? {
        return value
    }
}

private class ScanExec(private val ds: DataSource, private val projection: List<String>) : PhysicalPlan {
    override fun schema(): Schema {
        return ds.schema().select(projection)
    }

    override fun execute(): Sequence<RecordBatch> {
        return ds.scan(projection)
    }

    override fun children(): List<PhysicalPlan> {
        return listOf1()
    }

    override fun toString(): String {
        return "ScanExec: schema=${schema()}, projection=$projection"
    }
}

private class ProjectionExec(
    private val input: PhysicalPlan, val schema: Schema, private val expr: List<Expression>
) : PhysicalPlan {
    override fun schema(): Schema {
        return schema
    }

    override fun execute(): Sequence<RecordBatch> {
        return input.execute().map { batch ->
            val columns = expr.map { it.evaluate(batch) }
            RecordBatch(schema, columns)
        }
    }

    override fun children(): List<PhysicalPlan> {
        return listOf1(input)
    }

    override fun toString(): String {
        return "ProjectionExec: $expr"
    }
}

private class HashAggregateExec(
    private val input: PhysicalPlan,
    private val groupExpr: List<Expression>,
    private val aggregateExpr: List<AggregateExpression>,
    val schema: Schema
) : PhysicalPlan {
    override fun schema(): Schema {
        return schema
    }

    override fun execute(): Sequence<RecordBatch> {
        val map = HashMap<List<Any?>, List<Accumulator>>()
        input.execute().iterator().forEach { batch ->
            val groupKeys = groupExpr.map { it.evaluate(batch) }
            val aggrInputValues = aggregateExpr.map { it.inputExpression().evaluate(batch) }
            (0..<batch.rowCount()).forEach { rowIndex ->
                val rowKey = groupKeys.map {
                    when (val value = it.getValue(rowIndex)) {
                        is ByteArray -> String(value)
                        else -> value
                    }
                }
                val accumulators = map.getOrPut(rowKey) { aggregateExpr.map { it.createAccumulator() } }
                accumulators.withIndex().forEach { accum ->
                    val value = aggrInputValues[accum.index].getValue(rowIndex)
                    accum.value.accumulate(value)
                }
            }

        }
        val root = VectorSchemaRoot.create(schema.toArrow(), RootAllocator(Long.MAX_VALUE))
        root.allocateNew()
        root.rowCount = map.size
        val builders = root.fieldVectors.map { ArrowVectorBuilder(it) }
        map.entries.withIndex().forEach { entry ->
            val rowIndex = entry.index
            val groupingKey = entry.value.key
            val accumulators = entry.value.value
            groupExpr.indices.forEach { builders[it].set(rowIndex, groupingKey[it]) }
            aggregateExpr.indices.forEach {
                builders[groupExpr.size + it].set(rowIndex, accumulators[it].finalValue())
            }
        }

        val outputBatch = RecordBatch(schema, root.fieldVectors.map { ArrowFieldVector(it) })
        // println("HashAggregateExec output: \n${outputBatch.toCSV()}")
        return listOf1(outputBatch).asSequence()
    }

    override fun children(): List<PhysicalPlan> {
        return listOf1(input)
    }

    override fun toString(): String {
        return "HashAggregateExec: groupExpr=$groupExpr, aggrExpr=$aggregateExpr"
    }
}

private fun createPhysicalExpr(expr: LogicalExpr, input: LogicalPlan): Expression = when (expr) {
    is Column -> {
        val i = input.schema().fields.indexOfFirst { it.name == expr.name }
        if (i == -1) {
            throw SQLException("No column named '${expr.name}")
        }
        ColumnExpression(i)
    }

    is ColumnIndex -> ColumnExpression(expr.i)
    is Alias -> {
        createPhysicalExpr(expr.expr, input)
    }

    is CastExpr -> CastExpression(createPhysicalExpr(expr.expr, input), expr.dataType)
    else -> throw IllegalStateException("Unknown expr: $expr")
}

private fun createPhysicalPlan(plan: LogicalPlan): PhysicalPlan {
    return when (plan) {
        is Scan -> ScanExec(plan.dataSource, plan.projection)
        is Projection -> {
            val input = createPhysicalPlan(plan.input)
            val projectionExpr = plan.expr.map { createPhysicalExpr(it, plan.input) }
            val projectionSchema = Schema(plan.expr.map { it.toField(plan.input) })
            ProjectionExec(input, projectionSchema, projectionExpr)
        }

        is Aggregate -> {
            val input = createPhysicalPlan(plan.input)
            val groupExpr = plan.groupExpr.map { createPhysicalExpr(it, plan.input) }
            val aggregateExpr = plan.aggExpr.map {
                when (it) {
                    is Max -> MaxExpression(createPhysicalExpr(it.expr, plan.input))
                    else -> throw IllegalStateException(
                        "Unsupported aggregate function: $it"
                    )
                }
            }
            HashAggregateExec(input, groupExpr, aggregateExpr, plan.schema())
        }

        else -> throw IllegalStateException("Unknown physical plan")
    }
}

private interface OptimizerRule {
    fun optimize(plan: LogicalPlan): LogicalPlan
}

private fun extractColumns(
    expr: List<LogicalExpr>, input: LogicalPlan, accum: MutableSet<String>
) {
    expr.forEach { extractColumns(it, input, accum) }
}

private fun extractColumns(expr: LogicalExpr, input: LogicalPlan, accum: MutableSet<String>) {
    when (expr) {
        is Column -> accum.add(expr.name)
        is ColumnIndex -> {
            val element = input.schema().fields[expr.i].name
            accum.add(element)
        }

        is Alias -> {
            extractColumns(expr.expr, input, accum)
        }

        is CastExpr -> extractColumns(expr.expr, input, accum)
        is AggregateExpr -> {
            accum.add("fare_amount")
        }

        else -> throw IllegalStateException("extractColumns does not support expression: $expr")
    }
}

private class ProjectionPushDownRule : OptimizerRule {
    override fun optimize(plan: LogicalPlan): LogicalPlan {
        return pushDown(plan, mutableSetOf())
    }

    private fun pushDown(
        plan: LogicalPlan, columnNames: MutableSet<String>
    ): LogicalPlan {
        return when (plan) {
            is Projection -> {
                extractColumns(plan.expr, plan.input, columnNames)
                val input = pushDown(plan.input, columnNames)
                Projection(input, plan.expr)
            }

            is Aggregate -> {
                extractColumns(plan.groupExpr, plan.input, columnNames)
                extractColumns(plan.aggExpr.map { it.expr }, plan.input, columnNames)
                val input = pushDown(plan.input, columnNames)
                Aggregate(input, plan.groupExpr, plan.aggExpr)
            }

            is Scan -> {
                val validFieldNames = plan.dataSource.schema().fields.map { it.name }.toSet()
                val pushDown = validFieldNames.filter { columnNames.contains(it) }.toSet().sorted()
                Scan(plan.path, plan.dataSource, pushDown)
            }

            else -> throw UnsupportedOperationException()
        }
    }
}

private class CastExpression(private val expr: Expression, private val dataType: ArrowType) : Expression {

    override fun toString(): String {
        return "CAST($expr AS $dataType)"
    }

    override fun evaluate(input: RecordBatch): ColumnVector {
        val value: ColumnVector = expr.evaluate(input)
        val fieldVector = FieldVectorFactory.create(dataType, input.rowCount())
        val builder = ArrowVectorBuilder(fieldVector)

        when (dataType) {
            ArrowTypes.DoubleType -> {
                (0..<value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue = when (vv) {
                            is ByteArray -> String(vv).toDouble()
                            is String -> vv.toDouble()
                            is Number -> vv.toDouble()
                            else -> throw IllegalStateException("Cannot cast value to Double: $vv")
                        }
                        builder.set(it, castValue)
                    }
                }
            }

            else -> throw IllegalStateException("Cast to $dataType is not supported")
        }

        builder.setValueCount(value.size())
        return builder.build()
    }
}

private enum class Keyword : SqlTokenizer.TokenType {
    AS,
    BY,
    CAST,
    DOUBLE,
    FROM,
    GROUP,
    MAX,
    ORDER,
    SELECT;

    companion object {
        private val keywords = entries.associateBy(Keyword::name)
        fun textOf(text: String) = keywords[text.uppercase()]
    }
}

enum class Symbol(val text: String) : SqlTokenizer.TokenType {

    LEFT_PAREN("("), RIGHT_PAREN(")"), COMMA(",");

    companion object {
        private val symbols = entries.associateBy(Symbol::text)
        private val symbolStartSet = entries.flatMap { s -> s.text.toList() }.toSet()
        fun textOf(text: String) = symbols[text]
        fun isSymbol(ch: Char): Boolean {
            return symbolStartSet.contains(ch)
        }

        fun isSymbolStart(ch: Char): Boolean {
            return isSymbol(ch)
        }
    }
}

private data class Token(
    val text: String, val type: SqlTokenizer.TokenType, val endOffset: Int
) {

    override fun toString(): String {
        val typeType = when (type) {
            is Keyword -> "Keyword"
            is Symbol -> "Symbol"
            is SqlTokenizer.Literal -> "Literal"
            else -> ""
        }
        return "Token(\"$text\", $typeType.$type, $endOffset)"
    }
}

private class SqlTokenizer(val sql: String) {
    private var offset = 0

    class TokenStream(private val tokens: List<Token>) {
        var i = 0

        fun peek(): Token? {
            return if (i < tokens.size) {
                tokens[i]
            } else {
                null
            }
        }

        fun next(): Token? {
            return if (i < tokens.size) {
                tokens[i++]
            } else {
                null
            }
        }

        fun consumeKeywords(s: List<String>): Boolean {
            val save = i
            s.forEach { keyword ->
                if (!consumeKeyword(keyword)) {
                    i = save
                    return false
                }
            }
            return true
        }

        fun consumeKeyword(s: String): Boolean {
            val peek = peek()
            return if (peek?.type is Keyword && peek.text == s) {
                i++
                true
            } else {
                false
            }
        }

        fun consumeTokenType(t: TokenType): Boolean {
            val peek = peek()
            return if (peek?.type == t) {
                i++
                true
            } else {
                false
            }
        }

        override fun toString(): String {
            return tokens.withIndex().joinToString(" ") { (index, token) ->
                if (index == i) {
                    "*$token"
                } else {
                    token.toString()
                }
            }
        }
    }

    fun tokenize(): TokenStream {
        var token = nextToken()
        val list = mutableListOf<Token>()
        while (token != null) {
            list.add(token)
            token = nextToken()
        }
        return TokenStream(list)
    }

    interface TokenType

    enum class Literal : TokenType {
        LONG, DOUBLE, STRING, IDENTIFIER;

        companion object {
            fun isNumberStart(ch: Char): Boolean {
                return ch.isDigit() || '.' == ch
            }

            fun isIdentifierStart(ch: Char): Boolean {
                return ch.isLetter()
            }

            fun isIdentifierPart(ch: Char): Boolean {
                return ch.isLetter() || ch.isDigit() || ch == '_'
            }

            fun isCharsStart(ch: Char): Boolean {
                return '\'' == ch || '"' == ch
            }
        }
    }

    private fun nextToken(): Token? {
        offset = skipWhitespace(offset)
        var token: Token? = null
        when {
            offset >= sql.length -> {
                return null
            }

            Literal.isIdentifierStart(sql[offset]) -> {
                token = scanIdentifier(offset)
                offset = token.endOffset
            }

            Literal.isNumberStart(sql[offset]) -> {
                token = scanNumber(offset)
                offset = token.endOffset
            }

            Symbol.isSymbolStart(sql[offset]) -> {
                token = scanSymbol(offset)
                offset = token.endOffset
            }

            Literal.isCharsStart(sql[offset]) -> {
                token = scanChars(offset, sql[offset])
                offset = token.endOffset
            }
        }
        return token
    }

    private fun skipWhitespace(startOffset: Int): Int {
        return sql.indexOfFirst(startOffset) { ch -> !ch.isWhitespace() }
    }

    private fun scanNumber(startOffset: Int): Token {
        var endOffset = if ('-' == sql[startOffset]) {
            sql.indexOfFirst(startOffset + 1) { ch -> !ch.isDigit() }
        } else {
            sql.indexOfFirst(startOffset) { ch -> !ch.isDigit() }
        }
        if (endOffset == sql.length) {
            return Token(sql.substring(startOffset, endOffset), Literal.LONG, endOffset)
        }
        val isFloat = '.' == sql[endOffset]
        if (isFloat) {
            endOffset = sql.indexOfFirst(endOffset + 1) { ch -> !ch.isDigit() }
        }
        return Token(sql.substring(startOffset, endOffset), if (isFloat) Literal.DOUBLE else Literal.LONG, endOffset)
    }

    private fun scanIdentifier(startOffset: Int): Token {
        if ('`' == sql[startOffset]) {
            val endOffset: Int = getOffsetUntilTerminatedChar('`', startOffset)
            return Token(sql.substring(offset, endOffset), Literal.IDENTIFIER, endOffset)
        }
        val endOffset = sql.indexOfFirst(startOffset) { ch -> !Literal.isIdentifierPart(ch) }
        val text: String = sql.substring(startOffset, endOffset)
        val tokenType: TokenType = Keyword.textOf(text) ?: Literal.IDENTIFIER
        return Token(text, tokenType, endOffset)
    }

    private fun getOffsetUntilTerminatedChar(terminatedChar: Char, startOffset: Int): Int {
        val offset = sql.indexOf(terminatedChar, startOffset)
        return if (offset != -1) offset else throw TokenizeException()
    }

    private fun scanSymbol(startOffset: Int): Token {
        var endOffset = sql.indexOfFirst(startOffset) { ch -> !Symbol.isSymbol(ch) }
        var text = sql.substring(offset, endOffset)
        var symbol: Symbol?
        while (null == Symbol.textOf(text).also { symbol = it }) {
            text = sql.substring(offset, --endOffset)
        }
        return Token(text, symbol ?: throw TokenizeException(), endOffset)
    }

    private fun scanChars(startOffset: Int, terminatedChar: Char): Token {
        val endOffset = getOffsetUntilTerminatedChar(terminatedChar, startOffset + 1)
        return Token(sql.substring(startOffset + 1, endOffset), Literal.STRING, endOffset + 1)
    }

    private inline fun CharSequence.indexOfFirst(startIndex: Int = 0, predicate: (Char) -> Boolean): Int {
        for (index in startIndex..<this.length) {
            if (predicate(this[index])) {
                return index
            }
        }
        return sql.length
    }
}

private class TokenizeException : Throwable()

private interface SqlExpr

private data class SqlIdentifier(val id: String) : SqlExpr {
    override fun toString() = id
}

private data class SqlFunction(val id: String, val args: List<SqlExpr>) : SqlExpr {
    override fun toString() = id
}

private data class SqlAlias(val expr: SqlExpr, val alias: SqlIdentifier) : SqlExpr
private data class SqlCast(val expr: SqlExpr, val dataType: SqlIdentifier) : SqlExpr
private data class SqlSort(val expr: SqlExpr, val asc: Boolean) : SqlExpr
private interface SqlRelation : SqlExpr

private data class SqlSelect(
    val projection: List<SqlExpr>,
    val groupBy: List<SqlExpr>,
    val orderBy: List<SqlExpr>,
    val tableName: String
) : SqlRelation

// Pratt
private class SqlParser(private val tokens: SqlTokenizer.TokenStream) {

    fun parse(precedence: Int = 0): SqlExpr? {
        var expr = parsePrefix() ?: return null
        while (precedence < nextPrecedence()) {
            expr = parseInfix(expr)
        }
        return expr
    }

    fun nextPrecedence(): Int {
        val token = tokens.peek() ?: return 0
        val precedence = when (token.type) {
            Keyword.AS -> 10
            Symbol.LEFT_PAREN -> 70
            else -> 0
        }
        return precedence
    }

    fun parsePrefix(): SqlExpr? {
        val token = tokens.next() ?: return null
        val expr = when (token.type) {
            Keyword.SELECT -> parseSelect()
            Keyword.CAST -> parseCast()
            Keyword.MAX -> SqlIdentifier(token.text)
            Keyword.DOUBLE -> SqlIdentifier(token.text)
            SqlTokenizer.Literal.IDENTIFIER -> SqlIdentifier(token.text)
            else -> throw IllegalStateException("Unexpected token $token")
        }
        return expr
    }

    fun parseInfix(left: SqlExpr): SqlExpr {
        val token = tokens.peek()!!
        val expr = when (token.type) {
            Keyword.AS -> {
                tokens.next()
                SqlAlias(left, parseIdentifier())
            }

            Symbol.LEFT_PAREN -> {
                if (left is SqlIdentifier) {
                    tokens.next()
                    val args = parseExprList()
                    assert(tokens.next()?.type == Symbol.RIGHT_PAREN)
                    SqlFunction(left.id, args)
                } else {
                    throw IllegalStateException("Unexpected LPAREN")
                }
            }

            else -> throw IllegalStateException("Unexpected infix token $token")
        }
        return expr
    }

    private fun parseOrder(): List<SqlSort> {
        val sortList = mutableListOf<SqlSort>()
        var sort = parseExpr()
        while (sort != null) {
            sort = when (sort) {
                is SqlIdentifier -> SqlSort(sort, true)
                is SqlSort -> sort
                else -> throw java.lang.IllegalStateException("Unexpected expression $sort after order by.")
            }
            sortList.add(sort)

            if (tokens.peek()?.type == Symbol.COMMA) {
                tokens.next()
            } else {
                break
            }
            sort = parseExpr()
        }
        return sortList
    }

    private fun parseCast(): SqlCast {
        assert(tokens.consumeTokenType(Symbol.LEFT_PAREN))
        val expr = parseExpr() ?: throw SQLException()
        val alias = expr as SqlAlias
        assert(tokens.consumeTokenType(Symbol.RIGHT_PAREN))
        return SqlCast(alias.expr, alias.alias)
    }

    private fun parseSelect(): SqlSelect {
        val projection = parseExprList()

        if (tokens.consumeKeyword("FROM")) {
            val table = parseExpr() as SqlIdentifier

            var groupBy: List<SqlExpr> = listOf1()
            if (tokens.consumeKeywords(listOf1("GROUP", "BY"))) {
                groupBy = parseExprList()
            }

            var orderBy: List<SqlExpr> = listOf1()
            if (tokens.consumeKeywords(listOf1("ORDER", "BY"))) {
                orderBy = parseOrder()
            }

            return SqlSelect(projection, groupBy, orderBy, table.id)
        } else {
            throw IllegalStateException("Expected FROM keyword, found ${tokens.peek()}")
        }
    }

    private fun parseExprList(): List<SqlExpr> {
        val list = mutableListOf<SqlExpr>()
        var expr = parseExpr()
        while (expr != null) {
            list.add(expr)
            if (tokens.peek()?.type == Symbol.COMMA) {
                tokens.next()
            } else {
                break
            }
            expr = parseExpr()
        }
        return list
    }

    private fun parseExpr() = parse(0)

    private fun parseIdentifier(): SqlIdentifier {
        val expr = parseExpr() ?: throw SQLException("Expected identifier, found EOF")
        return when (expr) {
            is SqlIdentifier -> expr
            else -> throw SQLException("Expected identifier, found $expr")
        }
    }
}

private class ColumnIndex(val i: Int) : LogicalExpr {

    override fun toField(input: LogicalPlan): Field {
        return input.schema().fields[i]
    }

    override fun toString(): String {
        return "#$i"
    }
}

private fun createDataFrame(select: SqlSelect, tables: Map<String, DataFrame>): DataFrame {
    val table = tables[select.tableName] ?: throw SQLException("No table named '${select.tableName}'")
    val projectionExpr = select.projection.map { createLogicalExpr(it, table) }
    val aggregateExprCount = projectionExpr.count { isAggregateExpr(it) }
    if (aggregateExprCount == 0 && select.groupBy.isNotEmpty()) {
        throw SQLException("GROUP BY without aggregate expressions is not supported")
    }
    var plan = table

    val projection = mutableListOf<LogicalExpr>()
    val aggrExpr = mutableListOf<AggregateExpr>()
    val numGroupCols = select.groupBy.size
    var groupCount = 0

    projectionExpr.forEach { expr ->
        when (expr) {
            is AggregateExpr -> {
                projection.add(ColumnIndex(numGroupCols + aggrExpr.size))
                aggrExpr.add(expr)
            }

            is Alias -> {
                projection.add(Alias(ColumnIndex(numGroupCols + aggrExpr.size), expr.alias))
                aggrExpr.add(expr.expr as AggregateExpr)
            }

            else -> {
                projection.add(ColumnIndex(groupCount))
                groupCount += 1
            }
        }
    }
    plan = planAggregateQuery(select, plan, aggrExpr)
    plan = plan.project(projection)
    return plan
}

private fun isAggregateExpr(expr: LogicalExpr): Boolean {
    return when (expr) {
        is AggregateExpr -> true
        is Alias -> expr.expr is AggregateExpr
        else -> false
    }
}

private fun planAggregateQuery(
    select: SqlSelect,
    df: DataFrame,
    aggregateExpr: List<AggregateExpr>
): DataFrame {
    val groupByExpr = select.groupBy.map { createLogicalExpr(it, df) }
    return df.aggregate(groupByExpr, aggregateExpr)
}

private fun createLogicalExpr(expr: SqlExpr, input: DataFrame): LogicalExpr {
    return when (expr) {
        is SqlIdentifier -> Column(expr.id)
        is SqlAlias -> Alias(createLogicalExpr(expr.expr, input), expr.alias.id)
        is SqlCast -> CastExpr(createLogicalExpr(expr.expr, input), parseDataType(expr.dataType.id))
        is SqlFunction -> when (expr.id) {
            "MAX" -> Max(createLogicalExpr(expr.args.first(), input))
            else -> throw SQLException("Invalid aggregate function: $expr")
        }

        else -> throw SQLException("Cannot create logical expression from sql expression: $expr")
    }
}

private fun parseDataType(id: String): ArrowType {
    return when (id) {
        "double" -> ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
        else -> throw SQLException("Invalid data type $id")
    }
}

private class InMemoryDataSource(val schema: Schema, val data: List<RecordBatch>) : DataSource {

    override fun schema(): Schema {
        return schema
    }

    override fun scan(projection: List<String>): Sequence<RecordBatch> {
        val projectionIndices = projection.map { name -> schema.fields.indexOfFirst { it.name == name } }
        return data.asSequence().map { batch ->
            RecordBatch(schema, projectionIndices.map { i -> batch.field(i) })
        }
    }
}

@OptIn(DelicateCoroutinesApi::class)
private fun main() {
    val start = System.currentTimeMillis()
    val deferred = (1..12).map { month ->
        GlobalScope.async {
            executePartialQuery(month)
        }
    }
    val results: List<RecordBatch> = runBlocking {
        deferred.flatMap { it.await() }
    }
    val duration = System.currentTimeMillis() - start
    println("Collected ${results.size} batches in $duration ms")

    val sql = "SELECT VendorID, MAX(max_amount) FROM tripdata GROUP BY VendorID ORDER BY max_amount"

    val ctx = ExecutionContext()
    ctx.registerDataSource("tripdata", InMemoryDataSource(results.first().schema, results))
    val df = ctx.sql(sql)
    val result = ctx.execute(df)

    printQueryResult(result)
}

private fun executePartialQuery(month: Int): List<RecordBatch> {
    val sql = "SELECT VendorID, MAX(CAST(fare_amount AS double)) AS max_amount FROM tripdata GROUP BY VendorID"
    val partitionStart = System.currentTimeMillis()
    val monthStr = String.format("%02d", month)
    val filename = "yc-$monthStr.csv"
    val ctx = ExecutionContext()
    ctx.registerCsv("tripdata", filename)
    val df = ctx.sql(sql)
    val result = ctx.execute(df).toList()
    val duration = System.currentTimeMillis() - partitionStart
    println("Query against month $month took $duration ms")
    return result
}

private fun printQueryResult(queryResult: Sequence<RecordBatch>) {
    queryResult.forEach { batch ->
        (0..<batch.rowCount()).forEach { idx ->
            batch.fields.forEach { field ->
                print(field.getValue(idx))
                print(" ")
            }
            println()
        }
    }
}