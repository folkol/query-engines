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
import java.util.*
import java.util.logging.Logger
import kotlin.NoSuchElementException
import kotlin.collections.ArrayList
import kotlin.collections.HashMap
import kotlin.collections.listOf as listOf1

private object ArrowTypes {
    val BooleanType = ArrowType.Bool()
    val Int8Type = ArrowType.Int(8, true)
    val Int16Type = ArrowType.Int(16, true)
    val Int32Type = ArrowType.Int(32, true)
    val Int64Type = ArrowType.Int(64, true)
    val FloatType = ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    val DoubleType = ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    val StringType = ArrowType.Utf8()
}

private interface ColumnVector {
    fun getType(): ArrowType
    fun getValue(i: Int): Any?
    fun size(): Int
}

private class LiteralValueVector(
    private val arrowType: ArrowType, val value: Any?, private val size: Int
) : ColumnVector {
    override fun getType(): ArrowType {
        return arrowType
    }

    override fun getValue(i: Int): Any? {
        if (i < 0 || i >= size) {
            throw IndexOutOfBoundsException()
        }
        return value
    }

    override fun size(): Int {
        return size
    }
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

    fun project(indices: List<Int>): Schema {
        return Schema(indices.map { fields[it] })
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

// A relation with a known schema
private interface LogicalPlan {
    fun schema(): Schema
    fun children(): List<LogicalPlan>
}

private fun format(plan: LogicalPlan, indent: Int = 0): String {
    val b = StringBuilder()
    0.rangeTo(indent).forEach { _ -> b.append('\t') }
    b.append(plan.toString()).append('\n')
    plan.children().forEach { b.append(format(it, indent + 1)) }
    return b.toString()
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

private class LiteralString(val str: String) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(str, ArrowTypes.StringType)
    }

    override fun toString(): String {
        return "'$str'"
    }
}

private class LiteralLong(val n: Long) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(n.toString(), ArrowTypes.Int64Type)
    }

    override fun toString(): String {
        return n.toString()
    }
}

private class LiteralDouble(val n: Double) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(n.toString(), ArrowTypes.DoubleType)
    }

    override fun toString(): String {
        return n.toString()
    }
}

private abstract class BinaryExpr(
    val name: String, private val op: String, val l: LogicalExpr, val r: LogicalExpr
) : LogicalExpr {
    override fun toString(): String {
        return "$l $op $r"
    }
}

private abstract class BooleanBinaryExpr(
    name: String, op: String, l: LogicalExpr, r: LogicalExpr
) : BinaryExpr(name, op, l, r) {
    override fun toField(input: LogicalPlan): Field {
        return Field(name, ArrowTypes.BooleanType)
    }
}

private class Eq(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("eq", "=", l, r)
private class Neq(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("neq", "!=", l, r)
private class Gt(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("gt", ">", l, r)
private class GtEq(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("gteq", ">=", l, r)
private class Lt(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("lt", "<", l, r)
private class LtEq(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("lteq", "<=", l, r)

private class And(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("and", "AND", l, r)
private class Or(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("or", "OR", l, r)

private abstract class MathExpr(
    name: String, op: String, l: LogicalExpr, r: LogicalExpr
) : BooleanBinaryExpr(name, op, l, r) {
    override fun toField(input: LogicalPlan): Field {
        return Field("mult", l.toField(input).dataType)
    }
}

private class Add(l: LogicalExpr, r: LogicalExpr) : MathExpr("add", "+", l, r)
private class Subtract(l: LogicalExpr, r: LogicalExpr) : MathExpr("subtract", "-", l, r)
private class Multiply(l: LogicalExpr, r: LogicalExpr) : MathExpr("mult", "*", l, r)
private class Divide(l: LogicalExpr, r: LogicalExpr) : MathExpr("div", "/", l, r)
private class Modulus(l: LogicalExpr, r: LogicalExpr) : MathExpr("mod", "%", l, r)

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

private class Sum(input: LogicalExpr) : AggregateExpr("SUM", input)
private class Min(input: LogicalExpr) : AggregateExpr("MIN", input)
private class Max(input: LogicalExpr) : AggregateExpr("MAX", input)
private class Avg(input: LogicalExpr) : AggregateExpr("AVG", input)

private class Count(input: LogicalExpr = LiteralLong(1)) : AggregateExpr("COUNT", input) {
    override fun toField(input: LogicalPlan): Field {
        return Field("COUNT", ArrowTypes.Int32Type)
    }

    override fun toString(): String {
        return "COUNT($expr)"
    }
}

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

private class Selection(
    val input: LogicalPlan, val expr: LogicalExpr
) : LogicalPlan {
    override fun schema(): Schema {
        return input.schema()
    }

    override fun children(): List<LogicalPlan> {
        return listOf1(input)
    }

    override fun toString(): String {
        return "Filter: $expr"
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

    override fun getType(): ArrowType {
        return when (field) {
            is BitVector -> ArrowTypes.BooleanType
            is TinyIntVector -> ArrowTypes.Int8Type
            is SmallIntVector -> ArrowTypes.Int16Type
            is IntVector -> ArrowTypes.Int32Type
            is BigIntVector -> ArrowTypes.Int64Type
            is Float4Vector -> ArrowTypes.FloatType
            is Float8Vector -> ArrowTypes.DoubleType
            is VarCharVector -> ArrowTypes.StringType
            else -> throw IllegalStateException()
        }
    }

    override fun getValue(i: Int): Any? {

        if (field.isNull(i)) {
            return null
        }

        return when (field) {
            is BitVector -> field.get(i) == 1
            is TinyIntVector -> field.get(i)
            is SmallIntVector -> field.get(i)
            is IntVector -> field.get(i)
            is BigIntVector -> field.get(i)
            is Float4Vector -> field.get(i)
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

                is TinyIntVector -> rows.withIndex().forEach { row ->
                    val valueStr = row.value.getValue(field.value.name, "").trim()
                    if (valueStr.isEmpty()) {
                        vector.setNull(row.index)
                    } else {
                        vector.set(row.index, valueStr.toByte())
                    }
                }

                is SmallIntVector -> rows.withIndex().forEach { row ->
                    val valueStr = row.value.getValue(field.value.name, "").trim()
                    if (valueStr.isEmpty()) {
                        vector.setNull(row.index)
                    } else {
                        vector.set(row.index, valueStr.toShort())
                    }
                }

                is IntVector -> rows.withIndex().forEach { row ->
                    val valueStr = row.value.getValue(field.value.name, "").trim()
                    if (valueStr.isEmpty()) {
                        vector.setNull(row.index)
                    } else {
                        vector.set(row.index, valueStr.toInt())
                    }
                }

                is BigIntVector -> rows.withIndex().forEach { row ->
                    val valueStr = row.value.getValue(field.value.name, "").trim()
                    if (valueStr.isEmpty()) {
                        vector.setNull(row.index)
                    } else {
                        vector.set(row.index, valueStr.toLong())
                    }
                }

                is Float4Vector -> rows.withIndex().forEach { row ->
                    val valueStr = row.value.getValue(field.value.name, "").trim()
                    if (valueStr.isEmpty()) {
                        vector.setNull(row.index)
                    } else {
                        vector.set(row.index, valueStr.toFloat())
                    }
                }

                is Float8Vector -> rows.withIndex().forEach { row ->
                    val valueStr = row.value.getValue(field.value.name, "")
                    if (valueStr.isEmpty()) {
                        vector.setNull(row.index)
                    } else {
                        vector.set(row.index, valueStr.toDouble())
                    }
                }

                else -> throw IllegalStateException("No support for reading CSV columns with data type $vector")
            }
            field.value.valueCount = rows.size
        }

        return RecordBatch(schema, root.fieldVectors.map { ArrowFieldVector(it) })
    }
}

/**
 * Simple CSV data source. If no schema is provided then it assumes that the first line contains
 * field names and that all values are strings.
 */
private class CsvDataSource(
    private val filename: String,
    private val hasHeaders: Boolean,
    private val batchSize: Int,
    val schema: Schema?,
) : DataSource {

    private val logger = Logger.getLogger(CsvDataSource::class.simpleName)

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
        logger.fine("scan() projection=$projection")

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
        // parser will close once the end of the reader is reached
        parser.beginParsing(file.inputStream().reader())
        parser.detectedFormat

        return ReaderAsSequence(readSchema, parser, batchSize)
    }

    private fun inferSchema(): Schema {
        logger.fine("inferSchema()")

        val file = File(filename)
        if (!file.exists()) {
            throw FileNotFoundException(file.absolutePath)
        }

        val parser = buildParser(defaultSettings())
        return file.inputStream().use {
            parser.beginParsing(it.reader())
            parser.detectedFormat

            parser.parseNext()
            // some delimiters cause sparse arrays, so remove null columns in the parsed header
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
    fun filter(expr: LogicalExpr): DataFrame
    fun aggregate(groupBy: List<LogicalExpr>, aggregateExpr: List<AggregateExpr>): DataFrame
    fun schema(): Schema
    fun logicalPlan(): LogicalPlan
}

private class DataFrameImpl(private val plan: LogicalPlan) : DataFrame {
    override fun project(expr: List<LogicalExpr>): DataFrame {
        return DataFrameImpl(Projection(plan, expr))
    }

    override fun filter(expr: LogicalExpr): DataFrame {
        return DataFrameImpl(Selection(plan, expr))
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

    /** Execute the logical plan represented by a DataFrame */
    fun execute(df: DataFrame): Sequence<RecordBatch> {
        return execute(df.logicalPlan())
    }

    /** Execute the provided logical plan */
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

private fun col(name: String) = Column(name)

private class Alias(val expr: LogicalExpr, val alias: String) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(alias, expr.toField(input).dataType)
    }

    override fun toString(): String {
        return "$expr as $alias"
    }
}

/** Convenience method to wrap the current expression in an alias using an infix operator */
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

private class LiteralLongExpression(val value: Long) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        return LiteralValueVector(
            ArrowTypes.Int64Type, value, input.rowCount()
        )
    }
}

private class LiteralDoubleExpression(val value: Double) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        return LiteralValueVector(
            ArrowTypes.DoubleType, value, input.rowCount()
        )
    }
}

private class LiteralStringExpression(val value: String) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        return LiteralValueVector(
            ArrowTypes.StringType, value, input.rowCount()
        )
    }

    override fun toString(): String {
        return value
    }
}

private abstract class BinaryExpression(val l: Expression, val r: Expression) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        val ll = l.evaluate(input)
        val rr = r.evaluate(input)
        assert(ll.size() == rr.size())
        if (ll.getType() != rr.getType()) {
            throw IllegalStateException(
                "Binary expression operands do not have the same type: " + "${ll.getType()} != ${rr.getType()}"
            )
        }
        return evaluate(ll, rr)
    }

    abstract fun evaluate(l: ColumnVector, r: ColumnVector): ColumnVector
}

private abstract class BooleanExpression(val l: Expression, val r: Expression) : Expression {

    override fun evaluate(input: RecordBatch): ColumnVector {
        val ll = l.evaluate(input)
        val rr = r.evaluate(input)
        assert(ll.size() == rr.size())
        if (ll.getType() != rr.getType()) {
            throw IllegalStateException(
                "Cannot compare values of different type: ${ll.getType()} != ${rr.getType()}"
            )
        }
        return compare(ll, rr)
    }

    private fun compare(l: ColumnVector, r: ColumnVector): ColumnVector {
        val v = BitVector("v", RootAllocator(Long.MAX_VALUE))
        v.allocateNew()
        (0..<l.size()).forEach {
            val value = evaluate(l.getValue(it), r.getValue(it), l.getType())
            v.set(it, if (value) 1 else 0)
        }
        v.valueCount = l.size()
        return ArrowFieldVector(v)
    }

    abstract fun evaluate(l: Any?, r: Any?, arrowType: ArrowType): Boolean
}

private class EqExpression(l: Expression, r: Expression) : BooleanExpression(l, r) {
    override fun evaluate(l: Any?, r: Any?, arrowType: ArrowType): Boolean {
        return when (arrowType) {
            ArrowTypes.Int8Type -> (l as Byte) == (r as Byte)
            ArrowTypes.Int16Type -> (l as Short) == (r as Short)
            ArrowTypes.Int32Type -> (l as Int) == (r as Int)
            ArrowTypes.Int64Type -> (l as Long) == (r as Long)
            ArrowTypes.FloatType -> (l as Float) == (r as Float)
            ArrowTypes.DoubleType -> (l as Double) == (r as Double)
            ArrowTypes.StringType -> l.toString() == r.toString()
            else -> throw IllegalStateException(
                "Unsupported data type in comparison expression: $arrowType"
            )
        }
    }

    override fun toString(): String {
        return "$l = $r"
    }
}

object FieldVectorFactory {

    fun create(arrowType: ArrowType, initialCapacity: Int): FieldVector {
        val rootAllocator = RootAllocator(Long.MAX_VALUE)
        val fieldVector: FieldVector = when (arrowType) {
            ArrowTypes.BooleanType -> BitVector("v", rootAllocator)
            ArrowTypes.Int8Type -> TinyIntVector("v", rootAllocator)
            ArrowTypes.Int16Type -> SmallIntVector("v", rootAllocator)
            ArrowTypes.Int32Type -> IntVector("v", rootAllocator)
            ArrowTypes.Int64Type -> BigIntVector("v", rootAllocator)
            ArrowTypes.FloatType -> Float4Vector("v", rootAllocator)
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
                } else if (value is ByteArray) {
                    fieldVector.set(i, value)
                } else {
                    fieldVector.set(i, value.toString().toByteArray())
                }
            }

            is TinyIntVector -> {
                if (value == null) {
                    fieldVector.setNull(i)
                } else if (value is Number) {
                    fieldVector.set(i, value.toByte())
                } else if (value is String) {
                    fieldVector.set(i, value.toByte())
                } else {
                    throw IllegalStateException()
                }
            }

            is SmallIntVector -> {
                if (value == null) {
                    fieldVector.setNull(i)
                } else if (value is Number) {
                    fieldVector.set(i, value.toShort())
                } else if (value is String) {
                    fieldVector.set(i, value.toShort())
                } else {
                    throw IllegalStateException()
                }
            }

            is IntVector -> {
                if (value == null) {
                    fieldVector.setNull(i)
                } else if (value is Number) {
                    fieldVector.set(i, value.toInt())
                } else if (value is String) {
                    fieldVector.set(i, value.toInt())
                } else {
                    throw IllegalStateException()
                }
            }

            is BigIntVector -> {
                if (value == null) {
                    fieldVector.setNull(i)
                } else if (value is Number) {
                    fieldVector.set(i, value.toLong())
                } else if (value is String) {
                    fieldVector.set(i, value.toLong())
                } else {
                    throw IllegalStateException()
                }
            }

            is Float4Vector -> {
                if (value == null) {
                    fieldVector.setNull(i)
                } else if (value is Number) {
                    fieldVector.set(i, value.toFloat())
                } else if (value is String) {
                    fieldVector.set(i, value.toFloat())
                } else {
                    throw IllegalStateException()
                }
            }

            is Float8Vector -> {
                if (value == null) {
                    fieldVector.setNull(i)
                } else if (value is Number) {
                    fieldVector.set(i, value.toDouble())
                } else if (value is String) {
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

private abstract class MathExpression(l: Expression, r: Expression) : BinaryExpression(l, r) {
    override fun evaluate(l: ColumnVector, r: ColumnVector): ColumnVector {
        val fieldVector = FieldVectorFactory.create(l.getType(), l.size())
        val builder = ArrowVectorBuilder(fieldVector)
        (0..<l.size()).forEach {
            val value = evaluate(l.getValue(it), r.getValue(it), l.getType())
            builder.set(it, value)
        }
        builder.setValueCount(l.size())
        return builder.build()
    }

    abstract fun evaluate(l: Any?, r: Any?, arrowType: ArrowType): Any?
}

private class AddExpression(l: Expression, r: Expression) : MathExpression(l, r) {
    override fun evaluate(l: Any?, r: Any?, arrowType: ArrowType): Any {
        return when (arrowType) {
            ArrowTypes.Int8Type -> (l as Byte) + (r as Byte)
            ArrowTypes.Int16Type -> (l as Short) + (r as Short)
            ArrowTypes.Int32Type -> (l as Int) + (r as Int)
            ArrowTypes.Int64Type -> (l as Long) + (r as Long)
            ArrowTypes.FloatType -> (l as Float) + (r as Float)
            ArrowTypes.DoubleType -> (l as Double) + (r as Double)
            else -> throw IllegalStateException("Unsupported data type in math expression: $arrowType")
        }
    }

    override fun toString(): String {
        return "$l + $r"
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

private class SumExpression(private val expr: Expression) : AggregateExpression {

    override fun inputExpression(): Expression {
        return expr
    }

    override fun createAccumulator(): Accumulator {
        return SumAccumulator()
    }

    override fun toString(): String {
        return "SUM($expr)"
    }
}

private class SumAccumulator : Accumulator {

    var value: Any? = null

    override fun accumulate(value: Any?) {
        if (value != null) {
            if (this.value == null) {
                this.value = value
            } else {
                when (val currentValue = this.value) {
                    is Byte -> this.value = currentValue + value as Byte
                    is Short -> this.value = currentValue + value as Short
                    is Int -> this.value = currentValue + value as Int
                    is Long -> this.value = currentValue + value as Long
                    is Float -> this.value = currentValue + value as Float
                    is Double -> this.value = currentValue + value as Double
                    else -> throw UnsupportedOperationException(
                        "MIN is not implemented for type: ${value.javaClass.name}"
                    )
                }
            }
        }
    }

    override fun finalValue(): Any? {
        return value
    }
}

private class CountExpression(private val expr: Expression) : AggregateExpression {

    override fun inputExpression(): Expression {
        return expr
    }

    override fun createAccumulator(): Accumulator {
        return CountAccumulator()
    }

    override fun toString(): String {
        return "COUNT($expr)"
    }
}

private class CountAccumulator : Accumulator {

    var value: Int = 0

    override fun accumulate(value: Any?) {
        if (value != null) {
            this.value++
        }
    }

    override fun finalValue(): Int {
        return value
    }
}

private class ScanExec(private val ds: DataSource, private val projection: List<String>) : PhysicalPlan {
    override fun schema(): Schema {
        println("getting schema with projection: $projection")
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

private class SelectionExec(
    private val input: PhysicalPlan, private val expr: Expression
) : PhysicalPlan {
    override fun schema(): Schema {
        return input.schema()
    }

    override fun execute(): Sequence<RecordBatch> {
        val input = input.execute()
        return input.map { batch ->
            val result = (expr.evaluate(batch) as ArrowFieldVector).field as BitVector
            val schema = batch.schema
            val columnCount = batch.schema.fields.size
            val filteredFields = (0..<columnCount).map {
                filter(batch.field(it), result)
            }
            val fields = filteredFields.map { ArrowFieldVector(it) }
            RecordBatch(schema, fields)
        }
    }

    override fun children(): List<PhysicalPlan> {
        return listOf1(input)
    }

    private fun filter(v: ColumnVector, selection: BitVector): FieldVector {
        val filteredVector = VarCharVector("v", RootAllocator(Long.MAX_VALUE))
        filteredVector.allocateNew()
        val builder = ArrowVectorBuilder(filteredVector)
        var count = 0
        (0..<selection.valueCount).forEach {
            if (selection.get(it) == 1) {
                builder.set(count, v.getValue(it))
                count++
            }
        }
        filteredVector.valueCount = count
        return filteredVector
    }

    override fun toString(): String {
        return "SelectionExec: $expr"
    }
}

//interface PhysicalAggregateExpression {
//    fun inputExpression(): Expression
//    fun createAccumulator(): Accumulator
//}

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
                // print(rowKey)
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
        // note that there is no physical expression for an alias since the alias
        // only affects the name using in the planning phase and not how the aliased
        // expression is executed
        createPhysicalExpr(expr.expr, input)
    }

    is LiteralLong -> LiteralLongExpression(expr.n)
    is LiteralDouble -> LiteralDoubleExpression(expr.n)
    is LiteralString -> LiteralStringExpression(expr.str)
    is CastExpr -> CastExpression(createPhysicalExpr(expr.expr, input), expr.dataType)
    is BinaryExpr -> {
        val l = createPhysicalExpr(expr.l, input)
        val r = createPhysicalExpr(expr.r, input)
        when (expr) {
            // comparison
            is Eq -> EqExpression(l, r)

            // boolean

            // math
            is Add -> AddExpression(l, r)

            else -> throw IllegalStateException("Unsupported binary expression: $expr")
        }
    }

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

        is Selection -> {
            val input = createPhysicalPlan(plan.input)
            val filterExpr = createPhysicalExpr(plan.expr, plan.input)
            SelectionExec(input, filterExpr)
        }

        is Aggregate -> {
            val input = createPhysicalPlan(plan.input)
            val groupExpr = plan.groupExpr.map { createPhysicalExpr(it, plan.input) }
            val aggregateExpr = plan.aggExpr.map {
                when (it) {
                    is Max -> MaxExpression(createPhysicalExpr(it.expr, plan.input))
                    is Sum -> SumExpression(createPhysicalExpr(it.expr, plan.input))
                    is Count -> CountExpression(createPhysicalExpr(it.expr, plan.input))
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

/** Format a logical plan in human-readable form */
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
            println("ColumnIndex, found: $element")
            accum.add(element)
        }

        is Alias -> {
            extractColumns(expr.expr, input, accum)
            println("Recursing into aliased expression: $expr")
        }

        is CastExpr -> extractColumns(expr.expr, input, accum)

        is LiteralString -> {}
        is LiteralDouble -> {}
        is LiteralLong -> {}
        is Eq -> {}
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

            is Selection -> {
                extractColumns(plan.expr, plan, columnNames)
                val input = pushDown(plan.input, columnNames)
                Selection(input, plan.expr)
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
            ArrowTypes.Int8Type -> {
                (0..<value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue = when (vv) {
                            is ByteArray -> String(vv).toByte()
                            is String -> vv.toByte()
                            is Number -> vv.toByte()
                            else -> throw IllegalStateException("Cannot cast value to Byte: $vv")
                        }
                        builder.set(it, castValue)
                    }
                }
            }

            ArrowTypes.Int16Type -> {
                (0..<value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue = when (vv) {
                            is ByteArray -> String(vv).toShort()
                            is String -> vv.toShort()
                            is Number -> vv.toShort()
                            else -> throw IllegalStateException("Cannot cast value to Short: $vv")
                        }
                        builder.set(it, castValue)
                    }
                }
            }

            ArrowTypes.Int32Type -> {
                (0..<value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue = when (vv) {
                            is ByteArray -> String(vv).toInt()
                            is String -> vv.toInt()
                            is Number -> vv.toInt()
                            else -> throw IllegalStateException("Cannot cast value to Int: $vv")
                        }
                        builder.set(it, castValue)
                    }
                }
            }

            ArrowTypes.Int64Type -> {
                (0..<value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue = when (vv) {
                            is ByteArray -> String(vv).toLong()
                            is String -> vv.toLong()
                            is Number -> vv.toLong()
                            else -> throw IllegalStateException("Cannot cast value to Long: $vv")
                        }
                        builder.set(it, castValue)
                    }
                }
            }

            ArrowTypes.FloatType -> {
                (0..<value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue = when (vv) {
                            is ByteArray -> String(vv).toFloat()
                            is String -> vv.toFloat()
                            is Number -> vv.toFloat()
                            else -> throw IllegalStateException("Cannot cast value to Float: $vv")
                        }
                        builder.set(it, castValue)
                    }
                }
            }

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

            ArrowTypes.StringType -> {
                (0..<value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        builder.set(it, vv.toString())
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

    /**
     * common
     */
    SCHEMA, DATABASE, TABLE, COLUMN, VIEW, INDEX, TRIGGER, PROCEDURE, TABLESPACE, FUNCTION, SEQUENCE, CURSOR, FROM, TO, OF, IF, ON, FOR, WHILE, DO, NO, BY, WITH, WITHOUT, TRUE, FALSE, TEMPORARY, TEMP, COMMENT,

    /**
     * create
     */
    CREATE, REPLACE, BEFORE, AFTER, INSTEAD, EACH, ROW, STATEMENT, EXECUTE, BITMAP, NOSORT, REVERSE, COMPILE,

    /**
     * alter
     */
    ALTER, ADD, MODIFY, RENAME, ENABLE, DISABLE, VALIDATE, USER, IDENTIFIED,

    /**
     * truncate
     */
    TRUNCATE,

    /**
     * drop
     */
    DROP, CASCADE,

    /**
     * insert
     */
    INSERT, INTO, VALUES,

    /**
     * update
     */
    UPDATE, SET,

    /**
     * delete
     */
    DELETE,

    /**
     * select
     */
    SELECT, DISTINCT, AS, CASE, WHEN, ELSE, THEN, END, LEFT, RIGHT, FULL, INNER, OUTER, CROSS, JOIN, USE, USING, NATURAL, WHERE, ORDER, ASC, DESC, GROUP, HAVING, UNION,

    /**
     * others
     */
    DECLARE, GRANT, FETCH, REVOKE, CLOSE, CAST, NEW, ESCAPE, LOCK, SOME, LEAVE, ITERATE, REPEAT, UNTIL, OPEN, OUT, INOUT, OVER, ADVISE, SIBLINGS, LOOP, EXPLAIN, DEFAULT, EXCEPT, INTERSECT, MINUS, PASSWORD, LOCAL, GLOBAL, STORAGE, DATA, COALESCE,

    /**
     * Types
     */
    CHAR, CHARACTER, VARYING, VARCHAR, VARCHAR2, INTEGER, INT, SMALLINT, DECIMAL, DEC, NUMERIC, FLOAT, REAL, DOUBLE, PRECISION, DATE, TIME, INTERVAL, BOOLEAN, BLOB,

    /**
     * Conditionals
     */
    AND, OR, XOR, IS, NOT, NULL, IN, BETWEEN, LIKE, ANY, ALL, EXISTS,

    /**
     * Functions
     */
    AVG, MAX, MIN, SUM, COUNT, GREATEST, LEAST, ROUND, TRUNC, POSITION, EXTRACT, LENGTH, CHAR_LENGTH, SUBSTRING, SUBSTR, INSTR, INITCAP, UPPER, LOWER, TRIM, LTRIM, RTRIM, BOTH, LEADING, TRAILING, TRANSLATE, CONVERT, LPAD, RPAD, DECODE, NVL,

    /**
     * Constraints
     */
    CONSTRAINT, UNIQUE, PRIMARY, FOREIGN, KEY, CHECK, REFERENCES;

    companion object {
        private val keywords = entries.associateBy(Keyword::name)
        fun textOf(text: String) = keywords[text.uppercase(Locale.getDefault())]
    }
}

enum class Symbol(val text: String) : SqlTokenizer.TokenType {

    LEFT_PAREN("("), RIGHT_PAREN(")"), LEFT_BRACE("{"), RIGHT_BRACE("}"), LEFT_BRACKET("["), RIGHT_BRACKET("]"), SEMI(";"), COMMA(
        ","
    ),
    DOT("."), DOUBLE_DOT(".."), PLUS("+"), SUB("-"), STAR("*"), SLASH("/"), QUESTION("?"), EQ("="), GT(">"), LT("<"), BANG(
        "!"
    ),
    TILDE("~"), CARET("^"), PERCENT("%"), COLON(":"), DOUBLE_COLON("::"), COLON_EQ(":="), LT_EQ("<="), GT_EQ(">="), LT_EQ_GT(
        "<=>"
    ),
    LT_GT("<>"), BANG_EQ("!="), BANG_GT("!>"), BANG_LT("!<"), AMP("&"), BAR("|"), DOUBLE_AMP("&&"), DOUBLE_BAR("||"), DOUBLE_LT(
        "<<"
    ),
    DOUBLE_GT(">>"), AT("@"), POUND("#");

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

        private val logger = Logger.getLogger(TokenStream::class.simpleName)

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
            logger.fine("consumeKeyword('$s') next token is $peek")
            return if (peek?.type is Keyword && peek.text == s) {
                i++
                logger.fine("consumeKeyword() returning true")
                true
            } else {
                logger.fine("consumeKeyword() returning false")
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

    /**
     * skip whitespace.
     * @return offset after whitespace skipped
     */
    private fun skipWhitespace(startOffset: Int): Int {
        return sql.indexOfFirst(startOffset) { ch -> !ch.isWhitespace() }
    }

    /**
     * scan number.
     *
     * @return number token
     */
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

    /**
     * scan identifier.
     *
     * @return identifier token
     */
    private fun scanIdentifier(startOffset: Int): Token {
        if ('`' == sql[startOffset]) {
            val endOffset: Int = getOffsetUntilTerminatedChar('`', startOffset)
            return Token(sql.substring(offset, endOffset), Literal.IDENTIFIER, endOffset)
        }
        val endOffset = sql.indexOfFirst(startOffset) { ch -> !Literal.isIdentifierPart(ch) }
        val text: String = sql.substring(startOffset, endOffset)
        return if (isAmbiguousIdentifier(text)) {
            Token(text, processAmbiguousIdentifier(endOffset, text), endOffset)
        } else {
            val tokenType: TokenType = Keyword.textOf(text) ?: Literal.IDENTIFIER
            Token(text, tokenType, endOffset)
        }
    }

    /**
     * table name: group / order
     * keyword: group by / order by
     *
     * @return
     */
    private fun isAmbiguousIdentifier(text: String): Boolean {
        return Keyword.ORDER.name.equals(text, true) || Keyword.GROUP.name.equals(text, true)
    }

    /**
     * process group by | order by
     */
    private fun processAmbiguousIdentifier(startOffset: Int, text: String): TokenType {
        val skipWhitespaceOffset = skipWhitespace(startOffset)
        return if (skipWhitespaceOffset != sql.length && Keyword.BY.name.equals(
                sql.substring(
                    skipWhitespaceOffset, skipWhitespaceOffset + 2
                ), true
            )
        ) Keyword.textOf(text)!! else Literal.IDENTIFIER
    }

    /**
     *  find another char's offset equals terminatedChar
     */
    private fun getOffsetUntilTerminatedChar(terminatedChar: Char, startOffset: Int): Int {
        val offset = sql.indexOf(terminatedChar, startOffset)
        return if (offset != -1) offset else throw TokenizeException()
    }

    /**
     * scan symbol.
     *
     * @return symbol token
     */
    private fun scanSymbol(startOffset: Int): Token {
        var endOffset = sql.indexOfFirst(startOffset) { ch -> !Symbol.isSymbol(ch) }
        var text = sql.substring(offset, endOffset)
        var symbol: Symbol?
        while (null == Symbol.textOf(text).also { symbol = it }) {
            text = sql.substring(offset, --endOffset)
        }
        return Token(text, symbol ?: throw TokenizeException(), endOffset)
    }

    /**
     * scan chars like 'abc' or "abc"
     */
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

/** Pratt Top Down Operator Precedence Parser. See https://tdop.github.io/ for paper. */
private interface PrattParser {

    /** Parse an expression */
    fun parse(precedence: Int = 0): SqlExpr? {
        var expr = parsePrefix() ?: return null
        while (precedence < nextPrecedence()) {
            expr = parseInfix(expr, nextPrecedence())
        }
        return expr
    }

    /** Get the precedence of the next token */
    fun nextPrecedence(): Int

    /** Parse the next prefix expression */
    fun parsePrefix(): SqlExpr?

    /** Parse the next infix expression */
    fun parseInfix(left: SqlExpr, precedence: Int): SqlExpr
}

/** Simple SQL identifier such as a table or column name */
private data class SqlIdentifier(val id: String) : SqlExpr {
    override fun toString() = id
}

/** Binary expression */
private data class SqlBinaryExpr(val l: SqlExpr, val op: String, val r: SqlExpr) : SqlExpr {
    override fun toString(): String = "$l $op $r"
}

/** SQL literal string */
private data class SqlString(val value: String) : SqlExpr {
    override fun toString() = "'$value'"
}

/** SQL literal long */
private data class SqlLong(val value: Long) : SqlExpr {
    override fun toString() = "$value"
}

/** SQL literal double */
private data class SqlDouble(val value: Double) : SqlExpr {
    override fun toString() = "$value"
}

/** SQL function call */
private data class SqlFunction(val id: String, val args: List<SqlExpr>) : SqlExpr {
    override fun toString() = id
}

/** SQL aliased expression */
private data class SqlAlias(val expr: SqlExpr, val alias: SqlIdentifier) : SqlExpr

private data class SqlCast(val expr: SqlExpr, val dataType: SqlIdentifier) : SqlExpr

private data class SqlSort(val expr: SqlExpr, val asc: Boolean) : SqlExpr

private interface SqlRelation : SqlExpr

private data class SqlSelect(
    val projection: List<SqlExpr>,
    val selection: SqlExpr?,
    val groupBy: List<SqlExpr>,
    val orderBy: List<SqlExpr>,
    val having: SqlExpr?,
    val tableName: String
) : SqlRelation

private class SqlParser(private val tokens: SqlTokenizer.TokenStream) : PrattParser {

    private val logger = Logger.getLogger(SqlParser::class.simpleName)

    override fun nextPrecedence(): Int {
        val token = tokens.peek() ?: return 0
        val precedence = when (token.type) {
            // Keywords
            Keyword.AS, Keyword.ASC, Keyword.DESC -> 10
            Keyword.OR -> 20
            Keyword.AND -> 30

            // Symbols
            Symbol.LT, Symbol.LT_EQ, Symbol.EQ, Symbol.BANG_EQ, Symbol.GT_EQ, Symbol.GT -> 40

            Symbol.PLUS, Symbol.SUB -> 50
            Symbol.STAR, Symbol.SLASH -> 60

            Symbol.LEFT_PAREN -> 70
            else -> 0
        }
        logger.fine("nextPrecedence($token) returning $precedence")
        return precedence
    }

    override fun parsePrefix(): SqlExpr? {
        logger.fine("parsePrefix() next token = ${tokens.peek()}")
        val token = tokens.next() ?: return null
        val expr = when (token.type) {
            // Keywords
            Keyword.SELECT -> parseSelect()
            Keyword.CAST -> parseCast()

            Keyword.MAX -> SqlIdentifier(token.text)

            // type
            Keyword.INT -> SqlIdentifier(token.text)
            Keyword.DOUBLE -> SqlIdentifier(token.text)

            // Literals
            SqlTokenizer.Literal.IDENTIFIER -> SqlIdentifier(token.text)
            SqlTokenizer.Literal.STRING -> SqlString(token.text)
            SqlTokenizer.Literal.LONG -> SqlLong(token.text.toLong())
            SqlTokenizer.Literal.DOUBLE -> SqlDouble(token.text.toDouble())
            else -> throw IllegalStateException("Unexpected token $token")
        }
        logger.fine("parsePrefix() returning $expr")
        return expr
    }

    override fun parseInfix(left: SqlExpr, precedence: Int): SqlExpr {
        logger.fine("parseInfix() next token = ${tokens.peek()}")
        val token = tokens.peek()!!
        val expr = when (token.type) {
            Symbol.PLUS, Symbol.SUB, Symbol.STAR, Symbol.SLASH, Symbol.EQ, Symbol.GT, Symbol.LT -> {
                tokens.next() // consume the token
                SqlBinaryExpr(
                    left, token.text, parse(precedence) ?: throw SQLException("Error parsing infix")
                )
            }

            // keywords
            Keyword.AS -> {
                tokens.next() // consume the token
                SqlAlias(left, parseIdentifier())
            }

            Keyword.AND, Keyword.OR -> {
                tokens.next() // consume the token
                SqlBinaryExpr(
                    left, token.text, parse(precedence) ?: throw SQLException("Error parsing infix")
                )
            }

            Keyword.ASC, Keyword.DESC -> {
                tokens.next()
                SqlSort(left, token.type == Keyword.ASC)
            }


            Symbol.LEFT_PAREN -> {
                if (left is SqlIdentifier) {
                    tokens.next() // consume the token
                    val args = parseExprList()
                    assert(tokens.next()?.type == Symbol.RIGHT_PAREN)
                    SqlFunction(left.id, args)
                } else {
                    throw IllegalStateException("Unexpected LPAREN")
                }
            }

            else -> throw IllegalStateException("Unexpected infix token $token")
        }
        logger.fine("parseInfix() returning $expr")
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

            // parse optional WHERE clause
            var filterExpr: SqlExpr? = null
            if (tokens.consumeKeyword("WHERE")) {
                filterExpr = parseExpr()
            }

            // parse optional GROUP BY clause
            var groupBy: List<SqlExpr> = listOf1()
            if (tokens.consumeKeywords(listOf1("GROUP", "BY"))) {
                groupBy = parseExprList()
            }

            // parse optional HAVING clause
            var havingExpr: SqlExpr? = null
            if (tokens.consumeKeyword("HAVING")) {
                havingExpr = parseExpr()
            }

            // parse optional ORDER BY clause
            var orderBy: List<SqlExpr> = listOf1()
            if (tokens.consumeKeywords(listOf1("ORDER", "BY"))) {
                orderBy = parseOrder()
            }

            return SqlSelect(projection, filterExpr, groupBy, orderBy, havingExpr, table.id)
        } else {
            throw IllegalStateException("Expected FROM keyword, found ${tokens.peek()}")
        }
    }

    private fun parseExprList(): List<SqlExpr> {
        logger.fine("parseExprList()")
        val list = mutableListOf<SqlExpr>()
        var expr = parseExpr()
        while (expr != null) {
            // logger.fine("parseExprList parsed $expr")
            list.add(expr)
            if (tokens.peek()?.type == Symbol.COMMA) {
                tokens.next()
            } else {
                break
            }
            expr = parseExpr()
        }
        logger.fine("parseExprList() returning $list")
        return list
    }

    private fun parseExpr() = parse(0)

    /**
     * Parse the next token as an identifier, throwing an exception if the next token is not an
     * identifier.
     */
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

    // get a reference to the data source
    val table = tables[select.tableName] ?: throw SQLException("No table named '${select.tableName}'")

    // translate projection sql expressions into logical expressions
    val projectionExpr = select.projection.map { createLogicalExpr(it, table) }

    // build a list of columns referenced in the projection
    val columnNamesInProjection = getReferencedColumns(projectionExpr)

    val aggregateExprCount = projectionExpr.count { isAggregateExpr(it) }
    if (aggregateExprCount == 0 && select.groupBy.isNotEmpty()) {
        throw SQLException("GROUP BY without aggregate expressions is not supported")
    }

    // does the filter expression reference anything not in the final projection?
    val columnNamesInSelection = getColumnsReferencedBySelection(select, table)

    var plan = table

    if (aggregateExprCount == 0) {
        return planNonAggregateQuery(
            select, plan, projectionExpr, columnNamesInSelection, columnNamesInProjection
        )
    } else {
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
        plan = planAggregateQuery(projectionExpr, select, columnNamesInSelection, plan, aggrExpr)
        plan = plan.project(projection)
        if (select.having != null) {
            plan = plan.filter(createLogicalExpr(select.having, plan))
        }
        return plan
    }
}

private fun isAggregateExpr(expr: LogicalExpr): Boolean {
    // TODO implement this correctly .. this just handles aggregates and aliased aggregates
    return when (expr) {
        is AggregateExpr -> true
        is Alias -> expr.expr is AggregateExpr
        else -> false
    }
}

private fun planNonAggregateQuery(
    select: SqlSelect,
    df: DataFrame,
    projectionExpr: List<LogicalExpr>,
    columnNamesInSelection: Set<String>,
    columnNamesInProjection: Set<String>
): DataFrame {

    var plan = df
    if (select.selection == null) {
        return plan.project(projectionExpr)
    }

    val missing = (columnNamesInSelection - columnNamesInProjection)
    val n = projectionExpr.size
    plan = plan.project(projectionExpr + missing.map { Column(it) })
    plan = plan.filter(createLogicalExpr(select.selection, plan))

    // drop the columns that were added for the selection
    val expr = (0..<n).map { i -> Column(plan.schema().fields[i].name) }
    plan = plan.project(expr)

    return plan
}

private fun planAggregateQuery(
    projectionExpr: List<LogicalExpr>,
    select: SqlSelect,
    columnNamesInSelection: Set<String>,
    df: DataFrame,
    aggregateExpr: List<AggregateExpr>
): DataFrame {
    var plan = df
    val projectionWithoutAggregates = projectionExpr.filterNot { it is AggregateExpr }

    if (select.selection != null) {

        val columnNamesInProjectionWithoutAggregates = getReferencedColumns(projectionWithoutAggregates)
        val missing = (columnNamesInSelection - columnNamesInProjectionWithoutAggregates)

        // because the selection references some columns that are not in the projection output we
        // need to create an interim projection that has the additional columns, and then we need
        // to remove them after the selection has been applied
        plan = plan.project(projectionWithoutAggregates + missing.map { Column(it) })
        plan = plan.filter(createLogicalExpr(select.selection, plan))
    }

    val groupByExpr = select.groupBy.map { createLogicalExpr(it, plan) }
    return plan.aggregate(groupByExpr, aggregateExpr)
}

private fun getColumnsReferencedBySelection(select: SqlSelect, table: DataFrame): Set<String> {
    val accumulator = mutableSetOf<String>()
    if (select.selection != null) {
        val filterExpr = createLogicalExpr(select.selection, table)
        visit(filterExpr, accumulator)
        val validColumnNames = table.schema().fields.map { it.name }
        accumulator.removeIf { name -> !validColumnNames.contains(name) }
    }
    return accumulator
}

private fun getReferencedColumns(exprs: List<LogicalExpr>): Set<String> {
    val accumulator = mutableSetOf<String>()
    exprs.forEach { visit(it, accumulator) }
    return accumulator
}

private fun visit(expr: LogicalExpr, accumulator: MutableSet<String>) {
    //        logger.info("BEFORE visit() $expr, accumulator=$accumulator")
    when (expr) {
        is Column -> accumulator.add(expr.name)
        is Alias -> visit(expr.expr, accumulator)
        is BinaryExpr -> {
            visit(expr.l, accumulator)
            visit(expr.r, accumulator)
        }

        is AggregateExpr -> visit(expr.expr, accumulator)
    }
    //        logger.info("AFTER visit() $expr, accumulator=$accumulator")
}

private fun createLogicalExpr(expr: SqlExpr, input: DataFrame): LogicalExpr {
    return when (expr) {
        is SqlIdentifier -> Column(expr.id)
        is SqlString -> LiteralString(expr.value)
        is SqlLong -> LiteralLong(expr.value)
        is SqlDouble -> LiteralDouble(expr.value)
        is SqlBinaryExpr -> {
            val l = createLogicalExpr(expr.l, input)
            val r = createLogicalExpr(expr.r, input)
            when (expr.op) {
                // comparison operators
                "=" -> Eq(l, r)
                "!=" -> Neq(l, r)
                ">" -> Gt(l, r)
                ">=" -> GtEq(l, r)
                "<" -> Lt(l, r)
                "<=" -> LtEq(l, r)
                // boolean operators
                "AND" -> And(l, r)
                "OR" -> Or(l, r)
                // math operators
                "+" -> Add(l, r)
                "-" -> Subtract(l, r)
                "*" -> Multiply(l, r)
                "/" -> Divide(l, r)
                "%" -> Modulus(l, r)
                else -> throw SQLException("Invalid operator ${expr.op}")
            }
        }
        // is SqlUnaryExpr -> when (expr.op) {
        // "NOT" -> Not(createLogicalExpr(expr.l, input))
        // }
        is SqlAlias -> Alias(createLogicalExpr(expr.expr, input), expr.alias.id)
        is SqlCast -> CastExpr(createLogicalExpr(expr.expr, input), parseDataType(expr.dataType.id))
        is SqlFunction -> when (expr.id) {
            "MIN" -> Min(createLogicalExpr(expr.args.first(), input))
            "MAX" -> Max(createLogicalExpr(expr.args.first(), input))
            "SUM" -> Sum(createLogicalExpr(expr.args.first(), input))
            "AVG" -> Avg(createLogicalExpr(expr.args.first(), input))
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

private fun executeQuery(path: String, month: Int, sql: String): List<RecordBatch> {
    val monthStr = String.format("%02d", month)
//    val filename = "$path/yc-$monthStr.csv"
    // VendorID MAX
    // 1 70.9
    // 2 70.0
    val filename = "$path/ych-$monthStr.csv"
    val ctx = ExecutionContext()
    ctx.registerCsv("tripdata", filename)
    val df = ctx.sql(sql)
    return ctx.execute(df).toList()
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
            val sql = "SELECT VendorID, MAX(CAST(fare_amount AS double)) AS max_amount FROM tripdata GROUP BY VendorID"
            val partitionStart = System.currentTimeMillis()
            val result = executeQuery(".", month, sql)
            val duration = System.currentTimeMillis() - partitionStart
            println("Query against month $month took $duration ms")
            result
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

private fun printQueryResult(queryResult: Sequence<RecordBatch>) {
    var isFirst1 = true
    queryResult.forEach { batch ->
        if (isFirst1) {
            isFirst1 = false
            val headers = batch.schema.fields.joinToString(" ") { it.name }
            println(headers)
        }
        (0..<batch.rowCount()).forEach { idx ->
            batch.fields.forEach { field ->
                print(field.getValue(idx))
                print(" ")
            }
            println()
        }
    }
}