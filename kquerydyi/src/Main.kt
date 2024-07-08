import com.univocity.parsers.csv.CsvParser
import com.univocity.parsers.csv.CsvParserSettings
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
import java.util.logging.Logger
import kotlin.collections.listOf as listOf1

object ArrowTypes {
    val BooleanType = ArrowType.Bool()
    val Int8Type = ArrowType.Int(8, true)
    val Int16Type = ArrowType.Int(16, true)
    val Int32Type = ArrowType.Int(32, true)
    val Int64Type = ArrowType.Int(64, true)
    val UInt8Type = ArrowType.Int(8, false)
    val UInt16Type = ArrowType.Int(16, false)
    val UInt32Type = ArrowType.Int(32, false)
    val UInt64Type = ArrowType.Int(64, false)
    val FloatType = ArrowType.FloatingPoint(FloatingPointPrecision.SINGLE)
    val DoubleType = ArrowType.FloatingPoint(FloatingPointPrecision.DOUBLE)
    val StringType = ArrowType.Utf8()
}

interface ColumnVector {
    fun getType(): ArrowType
    fun getValue(i: Int): Any?
    fun size(): Int
}

class LiteralValueVector(
    val arrowType: ArrowType, val value: Any?, val size: Int
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

data class Field(val name: String, val dataType: ArrowType) {
    fun toArrow(): org.apache.arrow.vector.types.pojo.Field {
        val fieldType = org.apache.arrow.vector.types.pojo.FieldType(true, dataType, null)
        return org.apache.arrow.vector.types.pojo.Field(name, fieldType, listOf1())
    }
}

data class Schema(val fields: List<Field>) {

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

class RecordBatch(val schema: Schema, val fields: List<ColumnVector>) {
    fun rowCount() = fields.first().size()
    fun columnCount() = fields.size
    fun field(i: Int): ColumnVector {
        return fields[i]
    }
}

interface DataSource {
    fun schema(): Schema
    fun scan(projection: List<String>): Sequence<RecordBatch>
}

// A relation with a known schema
interface LogicalPlan {
    fun schema(): Schema
    fun children(): List<LogicalPlan>
}

fun format(plan: LogicalPlan, indent: Int = 0): String {
    val b = StringBuilder()
    0.rangeTo(indent).forEach { b.append('\t') }
    b.append(plan.toString()).append('\n')
    plan.children().forEach() { b.append(format(it, indent + 1)) }
    return b.toString()
}

interface LogicalExpr {
    fun toField(input: LogicalPlan): Field
}

class Column(val name: String) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return input.schema().fields.find { it.name == name } ?: throw SQLException("No column named '$name'")
    }

    override fun toString(): String {
        return "#$name"
    }
}

class LiteralString(val str: String) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(str, ArrowTypes.StringType)
    }

    override fun toString(): String {
        return "'$str'"
    }
}

class LiteralLong(val n: Long) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(n.toString(), ArrowTypes.Int64Type)
    }

    override fun toString(): String {
        return n.toString()
    }
}

class LiteralDouble(val n: Double) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(n.toString(), ArrowTypes.DoubleType)
    }

    override fun toString(): String {
        return n.toString()
    }
}

abstract class BinaryExpr(
    val name: String, val op: String, val l: LogicalExpr, val r: LogicalExpr
) : LogicalExpr {
    override fun toString(): String {
        return "$l $op $r"
    }
}

abstract class BooleanBinaryExpr(
    name: String, op: String, l: LogicalExpr, r: LogicalExpr
) : BinaryExpr(name, op, l, r) {
    override fun toField(input: LogicalPlan): Field {
        return Field(name, ArrowTypes.BooleanType)
    }
}

class Eq(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("eq", "=", l, r)
class Neq(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("neq", "!=", l, r)
class Gt(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("gt", ">", l, r)
class GtEq(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("gteq", ">=", l, r)
class Lt(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("lt", "<", l, r)
class LtEq(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("lteq", "<=", l, r)

class And(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("and", "AND", l, r)
class Or(l: LogicalExpr, r: LogicalExpr) : BooleanBinaryExpr("or", "OR", l, r)

abstract class MathExpr(
    name: String, op: String, l: LogicalExpr, r: LogicalExpr
) : BooleanBinaryExpr(name, op, l, r) {
    override fun toField(input: LogicalPlan): Field {
        return Field("mult", l.toField(input).dataType)
    }
}

class Add(l: LogicalExpr, r: LogicalExpr) : MathExpr("add", "+", l, r)
class Subtract(l: LogicalExpr, r: LogicalExpr) : MathExpr("subtract", "-", l, r)
class Multiply(l: LogicalExpr, r: LogicalExpr) : MathExpr("mult", "*", l, r)
class Divide(l: LogicalExpr, r: LogicalExpr) : MathExpr("div", "/", l, r)
class Modulus(l: LogicalExpr, r: LogicalExpr) : MathExpr("mod", "%", l, r)

abstract class AggregateExpr(
    val name: String, val expr: LogicalExpr
) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(name, expr.toField(input).dataType)
    }

    override fun toString(): String {
        return "$name($expr)"
    }
}

class Sum(input: LogicalExpr) : AggregateExpr("SUM", input)
class Min(input: LogicalExpr) : AggregateExpr("MIN", input)
class Max(input: LogicalExpr) : AggregateExpr("MAX", input)
class Avg(input: LogicalExpr) : AggregateExpr("AVG", input)

class Count(input: LogicalExpr = LiteralLong(1)) : AggregateExpr("COUNT", input) {
    override fun toField(input: LogicalPlan): Field {
        return Field("COUNT", ArrowTypes.Int32Type)
    }

    override fun toString(): String {
        return "COUNT($expr)"
    }
}

class Scan(
    val path: String, val dataSource: DataSource, val projection: List<String>
) : LogicalPlan {
    val schema = deriveSchema()

    override fun schema(): Schema {
        return schema
    }

    private fun deriveSchema(): Schema {
        val schema = dataSource.schema()
        if (projection.isEmpty()) {
            return schema
        } else {
            return schema.select(projection)
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

class Projection(
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
            expr.map {
                it.toString()
            }.joinToString(",")
        }"
    }
}

class Selection(
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

class Aggregate(
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

class ReaderAsSequence(
    private val schema: Schema, private val parser: CsvParser, private val batchSize: Int
) : Sequence<RecordBatch> {
    override fun iterator(): Iterator<RecordBatch> {
        return ReaderIterator(schema, parser, batchSize)
    }
}

class ArrowFieldVector(val field: FieldVector) : ColumnVector {

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
            is BitVector -> if (field.get(i) == 1) true else false
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

class ReaderIterator(
    private val schema: Schema, private val parser: CsvParser, private val batchSize: Int
) : Iterator<RecordBatch> {

    private val logger = Logger.getLogger(CsvDataSource::class.simpleName)

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
            val vector = field.value
            when (vector) {
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
class CsvDataSource(
    val filename: String,
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
                Schema(headers.mapIndexed { i, _ -> Field("field_${i + 1}", ArrowTypes.StringType) })
            }

            parser.stopParsing()
            schema
        }
    }
}

interface DataFrame {
    fun project(expr: List<LogicalExpr>): DataFrame
    fun filter(expr: LogicalExpr): DataFrame
    fun aggregate(groupBy: List<LogicalExpr>, aggregateExpr: List<AggregateExpr>): DataFrame
    fun schema(): Schema
    fun logicalPlan(): LogicalPlan
}

class DataFrameImpl(private val plan: LogicalPlan) : DataFrame {
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
        return plan.schema();
    }

    override fun logicalPlan(): LogicalPlan {
        return plan
    }
}

class ExecutionContext {
    private val tables = mutableMapOf<String, DataFrame>()

    fun sql(sql: String): DataFrame {
        val tokens = SqlTokenizer(sql).tokenize()
        val ast = SqlParser(tokens).parse() as SqlSelect
        val df = createDataFrame(ast, tables)
        return DataFrameImpl(df.logicalPlan())
    }

    fun csv(filename: String): DataFrame {
        return DataFrameImpl(Scan(filename, CsvDataSource(filename, true, 1000, null), listOf1()))
    }

    fun register(tablename: String, df: DataFrame) {
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
    fun execute(plan: LogicalPlan): Sequence<RecordBatch> {
//        val optimizedPlan = Optimizer().optimize(plan)
//        val physicalPlan = QueryPlanner().createPhysicalPlan(optimizedPlan)
        val optimizedPlan = ProjectionPushDownRule().optimize(plan)
        val physicalPlan = createPhysicalPlan(optimizedPlan)
//        val physicalPlan = createPhysicalPlan(plan)
        return physicalPlan.execute()
    }
    /// fun parquet
}

class CastExpr(val expr: LogicalExpr, val dataType: ArrowType) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(expr.toField(input).name, dataType)
    }

    override fun toString(): String {
        return "CAST($expr AS $dataType)"
    }
}

fun cast(expr: LogicalExpr, dataType: ArrowType) = CastExpr(expr, dataType)

//abstract class BinaryExpr(
//    val name: String, val op: String, val l: LogicalExpr, val r: LogicalExpr
//) : LogicalExpr {
//
//    override fun toString(): String {
//        return "$l $op $r"
//    }
//}

// supported expression objects
fun col(name: String) = Column(name)
fun lit(value: String) = LiteralString(value)
fun lit(value: Long) = LiteralLong(value)
fun lit(value: Double) = LiteralDouble(value)

infix fun LogicalExpr.eq(rhs: LogicalExpr): LogicalExpr {
    return Eq(this, rhs)
}

infix fun LogicalExpr.neq(rhs: LogicalExpr): LogicalExpr {
    return Neq(this, rhs)
}

infix fun LogicalExpr.gt(rhs: LogicalExpr): LogicalExpr {
    return Gt(this, rhs)
}

infix fun LogicalExpr.gteq(rhs: LogicalExpr): LogicalExpr {
    return GtEq(this, rhs)
}

infix fun LogicalExpr.lt(rhs: LogicalExpr): LogicalExpr {
    return Lt(this, rhs)
}

infix fun LogicalExpr.lteq(rhs: LogicalExpr): LogicalExpr {
    return LtEq(this, rhs)
}

infix fun LogicalExpr.mult(rhs: LogicalExpr): LogicalExpr {
    return Multiply(this, rhs)
}

class Alias(val expr: LogicalExpr, val alias: String) : LogicalExpr {
    override fun toField(input: LogicalPlan): Field {
        return Field(alias, expr.toField(input).dataType)
    }

    override fun toString(): String {
        return "$expr as $alias"
    }
}

/** Convenience method to wrap the current expression in an alias using an infix operator */
infix fun LogicalExpr.alias(alias: String): Alias {
    return Alias(this, alias)
}

interface PhysicalPlan {
    fun schema(): Schema
    fun execute(): Sequence<RecordBatch>
    fun children(): List<PhysicalPlan>
}

interface Expression {
    fun evaluate(input: RecordBatch): ColumnVector
}

class ColumnExpression(val i: Int) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        return input.field(i)
    }

    override fun toString(): String {
        return "#$i"
    }
}

class LiteralLongExpression(val value: Long) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        return LiteralValueVector(
            ArrowTypes.Int64Type, value, input.rowCount()
        )
    }
}

class LiteralDoubleExpression(val value: Double) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        return LiteralValueVector(
            ArrowTypes.DoubleType, value, input.rowCount()
        )
    }
}

class LiteralStringExpression(val value: String) : Expression {
    override fun evaluate(input: RecordBatch): ColumnVector {
        return LiteralValueVector(
            ArrowTypes.StringType, value, input.rowCount()
        )
    }

    override fun toString(): String {
        return value
    }
}

abstract class BinaryExpression(val l: Expression, val r: Expression) : Expression {
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

abstract class BooleanExpression(val l: Expression, val r: Expression) : Expression {

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

    fun compare(l: ColumnVector, r: ColumnVector): ColumnVector {
        val v = BitVector("v", RootAllocator(Long.MAX_VALUE))
        v.allocateNew()
        (0 until l.size()).forEach {
            val value = evaluate(l.getValue(it), r.getValue(it), l.getType())
            v.set(it, if (value) 1 else 0)
        }
        v.valueCount = l.size()
        return ArrowFieldVector(v)
    }

    abstract fun evaluate(l: Any?, r: Any?, arrowType: ArrowType): Boolean
}

class EqExpression(l: Expression, r: Expression) : BooleanExpression(l, r) {
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

class ArrowVectorBuilder(val fieldVector: FieldVector) {

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

abstract class MathExpression(l: Expression, r: Expression) : BinaryExpression(l, r) {
    override fun evaluate(l: ColumnVector, r: ColumnVector): ColumnVector {
        val fieldVector = FieldVectorFactory.create(l.getType(), l.size())
        val builder = ArrowVectorBuilder(fieldVector)
        (0 until l.size()).forEach {
            val value = evaluate(l.getValue(it), r.getValue(it), l.getType())
            builder.set(it, value)
        }
        builder.setValueCount(l.size())
        return builder.build()
    }

    abstract fun evaluate(l: Any?, r: Any?, arrowType: ArrowType): Any?
}

class AddExpression(l: Expression, r: Expression) : MathExpression(l, r) {
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

interface AggregateExpression {
    fun inputExpression(): Expression
    fun createAccumulator(): Accumulator
}

interface Accumulator {
    fun accumulate(value: Any?)
    fun finalValue(): Any?
}

class MaxExpression(val expr: Expression) : AggregateExpression {
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

class MaxAccumulator : Accumulator {
    var value: Any? = null
    override fun accumulate(value: Any?) {
        if (value != null) {
            if (this.value == null) {
                this.value = value
            } else {
                val isMax = when (value) {
                    is Byte -> value > this.value as Byte
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

class SumExpression(private val expr: Expression) : AggregateExpression {

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

class SumAccumulator : Accumulator {

    var value: Any? = null

    override fun accumulate(value: Any?) {
        if (value != null) {
            if (this.value == null) {
                this.value = value
            } else {
                val currentValue = this.value
                when (currentValue) {
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

class CountExpression(private val expr: Expression) : AggregateExpression {

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

class CountAccumulator : Accumulator {

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

class ScanExec(val ds: DataSource, val projection: List<String>) : PhysicalPlan {
    override fun schema(): Schema {
        println("getting schema with projection: $projection")
//        val schema = ds.schema()
//        val projection = schema.fields.withIndex().filter { it.index
//        }
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

class ProjectionExec(
    val input: PhysicalPlan, val schema: Schema, val expr: List<Expression>
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

class SelectionExec(
    val input: PhysicalPlan, val expr: Expression
) : PhysicalPlan {
    override fun schema(): Schema {
        return input.schema()
    }

    override fun execute(): Sequence<RecordBatch> {
        val input = input.execute();
        return input.map { batch ->
            val result = (expr.evaluate(batch) as ArrowFieldVector).field as BitVector
            val schema = batch.schema
            val columnCount = batch.schema.fields.size
            val filteredFields = (0 until columnCount).map {
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
        (0 until selection.valueCount).forEach {
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

class HashAggregateExec(
    val input: PhysicalPlan,
    val groupExpr: List<Expression>,
    val aggregateExpr: List<AggregateExpression>,
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
            (0 until batch.rowCount()).forEach { rowIndex ->
                val rowKey = groupKeys.map {
                    val value = it.getValue(rowIndex)
                    when (value) {
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

fun createPhysicalExpr(expr: LogicalExpr, input: LogicalPlan): Expression = when (expr) {
    is Column -> {
        val i = input.schema().fields.indexOfFirst { it.name == expr.name }
        if (i == -1) {
            throw SQLException("No column named '${expr.name}")
        }
        ColumnExpression(i)
    }

    is LiteralLong -> LiteralLongExpression(expr.n)
    is LiteralDouble -> LiteralDoubleExpression(expr.n)
    is LiteralString -> LiteralStringExpression(expr.str)
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

fun createPhysicalPlan(plan: LogicalPlan): PhysicalPlan {
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
fun formatPhysical(plan: PhysicalPlan, indent: Int = 0): String {
    val b = StringBuilder()
    0.until(indent).forEach { b.append("\t") }
    b.append(plan.toString()).append("\n")
    plan.children().forEach { b.append(formatPhysical(it, indent + 1)) }
    return b.toString()
}

interface OptimizerRule {
    fun optimize(plan: LogicalPlan): LogicalPlan
}

fun extractColumns(
    expr: List<LogicalExpr>, input: LogicalPlan, accum: MutableSet<String>
) {
    expr.forEach { extractColumns(it, input, accum) }
}

fun extractColumns(expr: LogicalExpr, input: LogicalPlan, accum: MutableSet<String>) {
    when (expr) {
        is Column -> accum.add(expr.name)
        is LiteralString -> {}
        is LiteralDouble -> {}
        is LiteralLong -> {}
        is Eq -> {}
        is Max -> accum.add(expr.name)
        else -> throw IllegalStateException("extractColumns does not support expression: $expr")
    }
}

class ProjectionPushDownRule : OptimizerRule {
    override fun optimize(plan: LogicalPlan): LogicalPlan {
        return pushDown(plan, mutableSetOf())
    }

    private fun pushDown(
        plan: LogicalPlan,
        columnNames: MutableSet<String>
    ): LogicalPlan {
        return when (plan) {
            is Projection -> {
                extractColumns(plan.expr, plan, columnNames)
                val input = pushDown(plan.input, columnNames)
                Projection(input, plan.expr)
            }

            is Selection -> {
                extractColumns(plan.expr, plan, columnNames)
                val input = pushDown(plan.input, columnNames)
                Selection(input, plan.expr)
            }

            is Aggregate -> {
                extractColumns(plan.groupExpr, plan, columnNames)
                extractColumns(plan.aggExpr.map { it.expr }, plan, columnNames)
                val input = pushDown(plan.input, columnNames)
                Aggregate(input, plan.groupExpr, plan.aggExpr)
            }

            is Scan -> Scan(plan.path, plan.dataSource, columnNames.toList().sorted())
            else -> throw UnsupportedOperationException()
        }
    }
}

class CastExpression(val expr: Expression, val dataType: ArrowType) : Expression {

    override fun toString(): String {
        return "CAST($expr AS $dataType)"
    }

    override fun evaluate(input: RecordBatch): ColumnVector {
        val value: ColumnVector = expr.evaluate(input)
        val fieldVector = FieldVectorFactory.create(dataType, input.rowCount())
        val builder = ArrowVectorBuilder(fieldVector)

        when (dataType) {
            ArrowTypes.Int8Type -> {
                (0 until value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue =
                            when (vv) {
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
                (0 until value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue =
                            when (vv) {
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
                (0 until value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue =
                            when (vv) {
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
                (0 until value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue =
                            when (vv) {
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
                (0 until value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue =
                            when (vv) {
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
                (0 until value.size()).forEach {
                    val vv = value.getValue(it)
                    if (vv == null) {
                        builder.set(it, null)
                    } else {
                        val castValue =
                            when (vv) {
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
                (0 until value.size()).forEach {
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

//interface Token
//data class IdentifierToken(val s: String) : Token
//data class LiteralStringToken(val s: String) : Token
//data class LiteralLongToken(val s: String) : Token
//data class KeywordToken(val s: String) : Token
//data class OperatorToken(val s: String) : Token


//class Tokenizer(val sql: String) {
//    fun tokenize(): List<Token> {
//        return emptyList()
//    }
//}


enum class Keyword : SqlTokenizer.TokenType {

    /**
     * common
     */
    SCHEMA,
    DATABASE,
    TABLE,
    COLUMN,
    VIEW,
    INDEX,
    TRIGGER,
    PROCEDURE,
    TABLESPACE,
    FUNCTION,
    SEQUENCE,
    CURSOR,
    FROM,
    TO,
    OF,
    IF,
    ON,
    FOR,
    WHILE,
    DO,
    NO,
    BY,
    WITH,
    WITHOUT,
    TRUE,
    FALSE,
    TEMPORARY,
    TEMP,
    COMMENT,

    /**
     * create
     */
    CREATE,
    REPLACE,
    BEFORE,
    AFTER,
    INSTEAD,
    EACH,
    ROW,
    STATEMENT,
    EXECUTE,
    BITMAP,
    NOSORT,
    REVERSE,
    COMPILE,

    /**
     * alter
     */
    ALTER,
    ADD,
    MODIFY,
    RENAME,
    ENABLE,
    DISABLE,
    VALIDATE,
    USER,
    IDENTIFIED,

    /**
     * truncate
     */
    TRUNCATE,

    /**
     * drop
     */
    DROP,
    CASCADE,

    /**
     * insert
     */
    INSERT,
    INTO,
    VALUES,

    /**
     * update
     */
    UPDATE,
    SET,

    /**
     * delete
     */
    DELETE,

    /**
     * select
     */
    SELECT,
    DISTINCT,
    AS,
    CASE,
    WHEN,
    ELSE,
    THEN,
    END,
    LEFT,
    RIGHT,
    FULL,
    INNER,
    OUTER,
    CROSS,
    JOIN,
    USE,
    USING,
    NATURAL,
    WHERE,
    ORDER,
    ASC,
    DESC,
    GROUP,
    HAVING,
    UNION,

    /**
     * others
     */
    DECLARE,
    GRANT,
    FETCH,
    REVOKE,
    CLOSE,
    CAST,
    NEW,
    ESCAPE,
    LOCK,
    SOME,
    LEAVE,
    ITERATE,
    REPEAT,
    UNTIL,
    OPEN,
    OUT,
    INOUT,
    OVER,
    ADVISE,
    SIBLINGS,
    LOOP,
    EXPLAIN,
    DEFAULT,
    EXCEPT,
    INTERSECT,
    MINUS,
    PASSWORD,
    LOCAL,
    GLOBAL,
    STORAGE,
    DATA,
    COALESCE,

    /**
     * Types
     */
    CHAR,
    CHARACTER,
    VARYING,
    VARCHAR,
    VARCHAR2,
    INTEGER,
    INT,
    SMALLINT,
    DECIMAL,
    DEC,
    NUMERIC,
    FLOAT,
    REAL,
    DOUBLE,
    PRECISION,
    DATE,
    TIME,
    INTERVAL,
    BOOLEAN,
    BLOB,

    /**
     * Conditionals
     */
    AND,
    OR,
    XOR,
    IS,
    NOT,
    NULL,
    IN,
    BETWEEN,
    LIKE,
    ANY,
    ALL,
    EXISTS,

    /**
     * Functions
     */
    AVG,
    MAX,
    MIN,
    SUM,
    COUNT,
    GREATEST,
    LEAST,
    ROUND,
    TRUNC,
    POSITION,
    EXTRACT,
    LENGTH,
    CHAR_LENGTH,
    SUBSTRING,
    SUBSTR,
    INSTR,
    INITCAP,
    UPPER,
    LOWER,
    TRIM,
    LTRIM,
    RTRIM,
    BOTH,
    LEADING,
    TRAILING,
    TRANSLATE,
    CONVERT,
    LPAD,
    RPAD,
    DECODE,
    NVL,

    /**
     * Constraints
     */
    CONSTRAINT,
    UNIQUE,
    PRIMARY,
    FOREIGN,
    KEY,
    CHECK,
    REFERENCES;

    companion object {
        private val keywords = values().associateBy(Keyword::name)
        fun textOf(text: String) = keywords[text.toUpperCase()]
    }
}

enum class Symbol(val text: String) : SqlTokenizer.TokenType {

    LEFT_PAREN("("),
    RIGHT_PAREN(")"),
    LEFT_BRACE("{"),
    RIGHT_BRACE("}"),
    LEFT_BRACKET("["),
    RIGHT_BRACKET("]"),
    SEMI(";"),
    COMMA(","),
    DOT("."),
    DOUBLE_DOT(".."),
    PLUS("+"),
    SUB("-"),
    STAR("*"),
    SLASH("/"),
    QUESTION("?"),
    EQ("="),
    GT(">"),
    LT("<"),
    BANG("!"),
    TILDE("~"),
    CARET("^"),
    PERCENT("%"),
    COLON(":"),
    DOUBLE_COLON("::"),
    COLON_EQ(":="),
    LT_EQ("<="),
    GT_EQ(">="),
    LT_EQ_GT("<=>"),
    LT_GT("<>"),
    BANG_EQ("!="),
    BANG_GT("!>"),
    BANG_LT("!<"),
    AMP("&"),
    BAR("|"),
    DOUBLE_AMP("&&"),
    DOUBLE_BAR("||"),
    DOUBLE_LT("<<"),
    DOUBLE_GT(">>"),
    AT("@"),
    POUND("#");

    companion object {
        private val symbols = values().associateBy(Symbol::text)
        private val symbolStartSet = values().flatMap { s -> s.text.toList() }.toSet()
        fun textOf(text: String) = symbols[text]
        fun isSymbol(ch: Char): Boolean {
            return symbolStartSet.contains(ch)
        }

        fun isSymbolStart(ch: Char): Boolean {
            return isSymbol(ch)
        }
    }
}

data class Token(
    val text: String,
    val type: SqlTokenizer.TokenType,
    val endOffset: Int
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

class SqlTokenizer(val sql: String) {

    // TODO this whole class is pretty crude and needs a lot of attention + unit tests (Hint: this
    // would be a great
    // place to start contributing!)

    var offset = 0

    class TokenStream(val tokens: List<Token>) {

        private val logger = Logger.getLogger(TokenStream::class.simpleName)

        var i = 0

        fun peek(): Token? {
            if (i < tokens.size) {
                return tokens[i]
            } else {
                return null
            }
        }

        fun next(): Token? {
            if (i < tokens.size) {
                return tokens[i++]
            } else {
                return null
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
            return tokens.withIndex()
                .map { (index, token) ->
                    if (index == i) {
                        "*$token"
                    } else {
                        token.toString()
                    }
                }
                .joinToString(" ")
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
        LONG,
        DOUBLE,
        STRING,
        IDENTIFIER;

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
                return token
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
        return if (skipWhitespaceOffset != sql.length
            && Keyword.BY.name.equals(sql.substring(skipWhitespaceOffset, skipWhitespaceOffset + 2), true)
        )
            Keyword.textOf(text)!! else Literal.IDENTIFIER
    }

    /**
     *  find another char's offset equals terminatedChar
     */
    private fun getOffsetUntilTerminatedChar(terminatedChar: Char, startOffset: Int): Int {
        val offset = sql.indexOf(terminatedChar, startOffset)
        return if (offset != -1) offset else
            throw TokenizeException("Must contain $terminatedChar in remain sql[$startOffset .. end]")
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
        return Token(text, symbol ?: throw TokenizeException("$text Must be a Symbol!"), endOffset)
    }

    /**
     * scan chars like 'abc' or "abc"
     */
    private fun scanChars(startOffset: Int, terminatedChar: Char): Token {
        val endOffset = getOffsetUntilTerminatedChar(terminatedChar, startOffset + 1)
        return Token(sql.substring(startOffset + 1, endOffset), Literal.STRING, endOffset + 1)
    }

    private inline fun CharSequence.indexOfFirst(startIndex: Int = 0, predicate: (Char) -> Boolean): Int {
        for (index in startIndex until this.length) {
            if (predicate(this[index])) {
                return index
            }
        }
        return sql.length
    }
}

class TokenizeException(val msg: String) : Throwable()

interface SqlExpr

/** Pratt Top Down Operator Precedence Parser. See https://tdop.github.io/ for paper. */
interface PrattParser {

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
data class SqlIdentifier(val id: String) : SqlExpr {
    override fun toString() = id
}

/** Binary expression */
data class SqlBinaryExpr(val l: SqlExpr, val op: String, val r: SqlExpr) : SqlExpr {
    override fun toString(): String = "$l $op $r"
}

/** SQL literal string */
data class SqlString(val value: String) : SqlExpr {
    override fun toString() = "'$value'"
}

/** SQL literal long */
data class SqlLong(val value: Long) : SqlExpr {
    override fun toString() = "$value"
}

/** SQL literal double */
data class SqlDouble(val value: Double) : SqlExpr {
    override fun toString() = "$value"
}

/** SQL function call */
data class SqlFunction(val id: String, val args: List<SqlExpr>) : SqlExpr {
    override fun toString() = id
}

/** SQL aliased expression */
data class SqlAlias(val expr: SqlExpr, val alias: SqlIdentifier) : SqlExpr

data class SqlCast(val expr: SqlExpr, val dataType: SqlIdentifier) : SqlExpr

data class SqlSort(val expr: SqlExpr, val asc: Boolean) : SqlExpr

interface SqlRelation : SqlExpr

data class SqlSelect(
    val projection: List<SqlExpr>,
    val selection: SqlExpr?,
    val groupBy: List<SqlExpr>,
    val orderBy: List<SqlExpr>,
    val having: SqlExpr?,
    val tableName: String
) : SqlRelation

class SqlParser(val tokens: SqlTokenizer.TokenStream) : PrattParser {

    private val logger = Logger.getLogger(SqlParser::class.simpleName)

    override fun nextPrecedence(): Int {
        val token = tokens.peek() ?: return 0
        val precedence =
            when (token.type) {
                // Keywords
                Keyword.AS, Keyword.ASC, Keyword.DESC -> 10
                Keyword.OR -> 20
                Keyword.AND -> 30

                // Symbols
                Symbol.LT, Symbol.LT_EQ, Symbol.EQ,
                Symbol.BANG_EQ, Symbol.GT_EQ, Symbol.GT -> 40

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
        val expr =
            when (token.type) {
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
        val expr =
            when (token.type) {
                Symbol.PLUS, Symbol.SUB, Symbol.STAR, Symbol.SLASH,
                Symbol.EQ, Symbol.GT, Symbol.LT -> {
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
                        left,
                        token.text,
                        parse(precedence) ?: throw SQLException("Error parsing infix")
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

private fun createLogicalExpr(expr: SqlExpr, input: DataFrame): LogicalExpr {
    return when (expr) {
        is SqlIdentifier -> Column(expr.id)
        is SqlAlias -> Alias(createLogicalExpr(expr.expr, input), expr.alias.id)
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
        is SqlFunction ->
            when (expr.id) {
                "MIN" -> Min(createLogicalExpr(expr.args.first(), input))
                "MAX" -> Max(createLogicalExpr(expr.args.first(), input))
                "SUM" -> Sum(createLogicalExpr(expr.args.first(), input))
                "AVG" -> Avg(createLogicalExpr(expr.args.first(), input))
                else -> throw SQLException("Invalid aggregate function: $expr")
            }

        else -> throw UnsupportedOperationException()
    }
}

private fun visit(expr: LogicalExpr, accumulator: MutableSet<String>) {
    when (expr) {
        is Column -> accumulator.add(expr.name)
        is Alias -> visit(expr.expr, accumulator)
        is BinaryExpr -> {
            visit(expr.l, accumulator)
            visit(expr.r, accumulator)
        }
    }
}

fun createDataFrame(select: SqlSelect, tables: Map<String, DataFrame>): DataFrame {

    // get a reference to the data source
    val df = tables[select.tableName] ?: throw SQLException("No table named '${select.tableName}'")

    // create the logical expressions for the projection
    val projectionExpr = select.projection.map { createLogicalExpr(it, df) }

    if (select.selection == null) {
        // if there is no selection then we can just return the projection
        return df.project(projectionExpr)
    }

    // create the logical expression to represent the selection
    val filterExpr = createLogicalExpr(select.selection, df)

    // get a list of columns references in the projection expression
    val columnsInProjection = projectionExpr
        .map { it.toField(df.logicalPlan()).name }
        .toSet()

    // get a list of columns referenced in the selection expression
    val columnNames = mutableSetOf<String>()
    visit(filterExpr, columnNames)

    // determine if the selection references any columns not in the projection
    val missing = columnNames - columnsInProjection

    // if the selection only references outputs from the projection we can
    // simply apply the filter expression to the DataFrame representing
    // the projection
    if (missing.size == 0) {
        return df.project(projectionExpr)
            .filter(filterExpr)
    }

    // because the selection references some columns that are not in the
    // projection output we need to create an interim projection that has
    // the additional columns and then we need to remove them after the
    // selection has been applied
    return df.project(projectionExpr + missing.map { Column(it) })
        .filter(filterExpr)
        .project(projectionExpr.map {
            Column(it.toField(df.logicalPlan()).name)
        })
}

private fun plan(sql: String): LogicalPlan {
    println("parse() $sql")

    val tokens = SqlTokenizer(sql).tokenize()
    println(tokens)

    val parsedQuery = SqlParser(tokens).parse()
    println(parsedQuery)

    val tables =
        mapOf(
            "employee" to
                    DataFrameImpl(Scan("", CsvDataSource("employee.csv", true, 1024, null), listOf1())),
            "yello" to
                    DataFrameImpl(Scan("", CsvDataSource("yellow_tripdata_2024-01.csv", true, 1024, null), listOf1()))
        )

    val df = createDataFrame(parsedQuery as SqlSelect, tables)

    val plan = df.logicalPlan()
    println(format(plan))

    return plan
}

fun executeQuery(path: String, month: Int, sql: String): List<RecordBatch> {
    val monthStr = String.format("%02d", month);
    val filename = "$path/yellow_tripdata_2024-$monthStr.csv"
    val ctx = ExecutionContext()
    ctx.registerCsv("tripdata", filename)
    val df = ctx.sql(sql)
    return ctx.execute(df).toList()
}

class InMemoryDataSource(val schema: Schema, val data: List<RecordBatch>) : DataSource {

    override fun schema(): Schema {
        return schema
    }

    override fun scan(projection: List<String>): Sequence<RecordBatch> {
        val projectionIndices =
            projection.map { name -> schema.fields.indexOfFirst { it.name == name } }
        return data.asSequence().map { batch ->
            RecordBatch(schema, projectionIndices.map { i -> batch.field(i) })
        }
    }
}

fun main() {

    // Here is some verbose code for building a plan for the query
    // SELECT * FROM employee WHERE state = 'CO'
    // against a CSV file containing the columns
    // id, first_name, last_name, state, job_title, salary
//    val csv = CsvDataSource("employee.csv", true, 10, null)
//    val scan = Scan("employee", csv, listOf())
//    val filterExpr = Eq(Column("state"), LiteralString("CO"))
//    val selection = Selection(scan, filterExpr)
//    val projectionList = listOf(
//        Column("id"),
//        Column("first_name"),
//        Column("last_name"),
//        Column("state"),
//        Column("salary"),
//    )
//    val plan = Projection(selection, projectionList)

//    val plan = Projection(
//        Selection(
//            Scan("employee", CsvDataSource("employee.csv", true, 10, null), listOf()),
//            Eq(Column("state"), LiteralString("CO"))
//        ), listOf(
//            Column("id"),
//            Column("first_name"),
//            Column("last_name"),
//            Column("state"),
//            Column("salary"),
//        )
//    )

//    val ctx = ExecutionContext()
//    val plan = ctx.csv("employee.csv")
//        .filter(Eq(Column("state"), LiteralString("CO")))
//        .project(
//            listOf(
//                Column("id"),
//                Column("first_name"),
//                Column("last_name"),
//                Column("state"),
//                Column("salary"),
//            )
//        )

//    val ctx = ExecutionContext()
//    val plan = ctx.csv("employee.csv")
//        .filter(Eq(col("state"), lit("CO")))
//        .project(
//            listOf(
//                col("id"),
//                col("first_name"),
//                col("last_name"),
//                col("state"),
//                col("salary"),
//            )
//        )
//
//    val ctx = ExecutionContext()
//    val plan = ctx.csv("employee.csv")
//        .filter(col("state") eq lit("CO"))
//        .project(
//            listOf(
//                col("id"),
//                col("first_name"),
//                col("last_name"),
//                col("state"),
//                col("salary"),
//                (col("salary") mult lit(0.1) alias "bonus")
//            )
//        ).filter(col("bonus") gt lit(1000))
//
//    println(format(plan.logicalPlan()))
//
//    val csvPlan = ctx.csv("employee.csv")
//        .filter(col("state") eq lit("Uppsala"))
//        .aggregate(
//            listOf(col("state")), listOf(Count())
//        )
//
//    val physicalPlan = createPhysicalPlan(csvPlan.logicalPlan())
//    println(formatPhysical(physicalPlan))
//    printQueryResult(physicalPlan.execute())
//
//    println("=== Optimized Plan ===")
//    val optimizedCsvPlan = ProjectionPushDownRule().optimize(csvPlan.logicalPlan())
//    val optimizedPhysicalPlan = createPhysicalPlan(optimizedCsvPlan)
//    println(formatPhysical(optimizedPhysicalPlan))
//    printQueryResult(optimizedPhysicalPlan.execute())

//    val yellowCab = ctx.csv("yellow_tripdata_2024-01.csv")
//        .filter(col("state") eq lit("Uppsala"))
//        .filter(col("Airport_fee") eq lit("0.0"))
//        .aggregate(
//            listOf1(col("passenger_count")), listOf1(Count())
//        )

//    (0..<3).forEach { doit(yellowCab) }
//    (0..<3).forEach { doitFast(yellowCab) }
//    doitFast(yellowCab)
//    println(formatPhysical(optimizedPhysicalPlan))
//    printQueryResult(optimizedPhysicalPlan.execute())


//    val sqlExpr = SqlBinaryExpr(SqlIdentifier("foo"), "=", SqlString("bar"))

//    val sql = "1 + 2 * 3";
//    val sql = "SELECT foo, bar FROM lullaby WHERE foo = bar";
//    println("parse() $sql")
//    val tokens = SqlTokenizer(sql).tokenize()
//    println(tokens)
//    val parsedQuery = SqlParser(tokens).parse()
//    println(parsedQuery)

    // id,first_name,last_name,state,job_title,salary
    // 1,Matte,Johansson,Uppsala,Engineer,1337
    // 2,Other,Person,Uppsala,Worker,666
    // 3,Pelle,Pärsson,Sthlm,Unemployed,0
//    val plan = plan("SELECT first_name FROM employee WHERE state = 'Uppsala'")
//    doitPlan(plan)
//    println(plan.toString())

//    val plan = plan("SELECT passenger_count FROM yello WHERE PULocationID = '148'")
//    doitPlan(plan)
//    println(plan.toString())

//    val result = executeQuery(".", 1, "SELECT PULocationID FROM tripdata")
//    println(result)

    val start = System.currentTimeMillis()
    val deferred = (1..2).map {month ->
        GlobalScope.async {

            val sql = "SELECT passenger_count FROM tripdata " +
                    "GROUP BY passenger_count"

            val start = System.currentTimeMillis()
            val result = executeQuery(".", month, sql)
            val duration = System.currentTimeMillis() - start
            println("Query against month $month took $duration ms")
            result
        }
    }
    val results: List<RecordBatch> = runBlocking {
        deferred.flatMap { it.await() }
    }
    val duration = System.currentTimeMillis() - start
    println("Collected ${results.size} batches in $duration ms")

    val sql = "SELECT passenger_count " +
            "FROM tripdata " +
            "GROUP BY passenger_count"

    val ctx = ExecutionContext()
    ctx.registerDataSource("tripdata", InMemoryDataSource(results.first().schema, results))
    val df = ctx.sql(sql)
    val result = ctx.execute(df)

    printQueryResult(result)
}

private fun doit(yellowCab: DataFrame) {
    val start = System.currentTimeMillis()
    println("going yello")
//    val optimizedYellowCab = ProjectionPushDownRule().optimize(yellowCab.logicalPlan())
    val optimizedPhysicalPlan = createPhysicalPlan(yellowCab.logicalPlan())
//    println(formatPhysical(optimizedPhysicalPlan))
    consumeQueryResult(optimizedPhysicalPlan.execute())
    val elapsed = System.currentTimeMillis() - start;
    println("done in $elapsed ms")
    printQueryResult(optimizedPhysicalPlan.execute())
}

private fun doitPlan(plan: LogicalPlan) {
    val start = System.currentTimeMillis()
    println("going plan brrrrrrr")
//    val optimizedYellowCab = ProjectionPushDownRule().optimize(yellowCab.logicalPlan())
    val optimizedPhysicalPlan = createPhysicalPlan(plan)
//    println(formatPhysical(optimizedPhysicalPlan))
    consumeQueryResult(optimizedPhysicalPlan.execute())
    val elapsed = System.currentTimeMillis() - start;
    println("done in $elapsed ms")
    printQueryResult(optimizedPhysicalPlan.execute())
}

private fun doitFast(yellowCab: DataFrame) {
    val start = System.currentTimeMillis()
    println("going yello fast")
    val optimizedYellowCab = ProjectionPushDownRule().optimize(yellowCab.logicalPlan())
    val optimizedPhysicalPlan = createPhysicalPlan(optimizedYellowCab)
//    println(formatPhysical(optimizedPhysicalPlan))
    val queryResult = optimizedPhysicalPlan.execute()
    consumeQueryResult(queryResult)
    val elapsed = System.currentTimeMillis() - start;
    println("done in $elapsed ms")
    printQueryResult(queryResult)
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

private fun consumeQueryResult(queryResult: Sequence<RecordBatch>) {
    val count = queryResult.count()
    println("got $count batches")
//    var isFirst1 = true
//    queryResult.forEach { batch ->
//        if (isFirst1) {
//            isFirst1 = false
//            val headers = batch.schema.fields.joinToString(" ") { it.name }
//            println(headers)
//        }
//        (0..<batch.rowCount()).forEach { idx ->
//            batch.fields.forEach { field ->
//                print(field.getValue(idx))
//                print(" ")
//            }
//            println()
//        }
//    }
}