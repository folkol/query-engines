import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType

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

fun main() {

}