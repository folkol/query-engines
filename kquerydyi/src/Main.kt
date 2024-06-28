import com.univocity.parsers.csv.CsvParser
import com.univocity.parsers.csv.CsvParserSettings
import org.apache.arrow.memory.RootAllocator
import org.apache.arrow.vector.*
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType
import java.io.File
import java.io.FileNotFoundException
import java.sql.SQLException
import java.util.logging.Logger

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
        return org.apache.arrow.vector.types.pojo.Field(name, fieldType, listOf())
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

class Count(input: LogicalExpr) : AggregateExpr("COUNT", input) {
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
        return listOf()
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
        return listOf(input)
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
        return listOf(input)
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
        return listOf(input)
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
        val root = VectorSchemaRoot.create(schema.toArrow(), RootAllocator(Long.MAX_VALUE))
        root.fieldVectors.forEach { it.setInitialCapacity(rows.size) }
        root.allocateNew()

        root.fieldVectors.withIndex().forEach { field ->
            val vector = field.value
            when (vector) {
                is VarCharVector ->
                    rows.withIndex().forEach { row ->
                        val valueStr = row.value.getValue(field.value.name, "").trim()
                        vector.setSafe(row.index, valueStr.toByteArray())
                    }

                is TinyIntVector ->
                    rows.withIndex().forEach { row ->
                        val valueStr = row.value.getValue(field.value.name, "").trim()
                        if (valueStr.isEmpty()) {
                            vector.setNull(row.index)
                        } else {
                            vector.set(row.index, valueStr.toByte())
                        }
                    }

                is SmallIntVector ->
                    rows.withIndex().forEach { row ->
                        val valueStr = row.value.getValue(field.value.name, "").trim()
                        if (valueStr.isEmpty()) {
                            vector.setNull(row.index)
                        } else {
                            vector.set(row.index, valueStr.toShort())
                        }
                    }

                is IntVector ->
                    rows.withIndex().forEach { row ->
                        val valueStr = row.value.getValue(field.value.name, "").trim()
                        if (valueStr.isEmpty()) {
                            vector.setNull(row.index)
                        } else {
                            vector.set(row.index, valueStr.toInt())
                        }
                    }

                is BigIntVector ->
                    rows.withIndex().forEach { row ->
                        val valueStr = row.value.getValue(field.value.name, "").trim()
                        if (valueStr.isEmpty()) {
                            vector.setNull(row.index)
                        } else {
                            vector.set(row.index, valueStr.toLong())
                        }
                    }

                is Float4Vector ->
                    rows.withIndex().forEach { row ->
                        val valueStr = row.value.getValue(field.value.name, "").trim()
                        if (valueStr.isEmpty()) {
                            vector.setNull(row.index)
                        } else {
                            vector.set(row.index, valueStr.toFloat())
                        }
                    }

                is Float8Vector ->
                    rows.withIndex().forEach { row ->
                        val valueStr = row.value.getValue(field.value.name, "")
                        if (valueStr.isEmpty()) {
                            vector.setNull(row.index)
                        } else {
                            vector.set(row.index, valueStr.toDouble())
                        }
                    }

                else ->
                    throw IllegalStateException("No support for reading CSV columns with data type $vector")
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

        val readSchema =
            if (projection.isNotEmpty()) {
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

            val schema =
                if (hasHeaders) {
                    Schema(headers.map { colName -> Field(colName, ArrowTypes.StringType) })
                } else {
                    Schema(headers.mapIndexed { i, _ -> Field("field_${i + 1}", ArrowTypes.StringType) })
                }

            parser.stopParsing()
            schema
        }
    }
}


fun main() {

    // Here is some verbose code for building a plan for the query
    // SELECT * FROM employee WHERE state = 'CO'
    // against a CSV file containing the columns
    // id, first_name, last_name, state, job_title, salary
    val csv = CsvDataSource("employee.csv", true, 10, null)
    val scan = Scan("employee", csv, listOf())
    val filterExpr = Eq(Column("state"), LiteralString("CO"))
    val selection = Selection(scan, filterExpr)
    val projectionList = listOf(
        Column("id"),
        Column("first_name"),
        Column("last_name"),
        Column("state"),
        Column("salary"),
    )
    val plan = Projection(selection, projectionList)

    println(format(plan))
}