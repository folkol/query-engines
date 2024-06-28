import com.fasterxml.jackson.databind.DatabindException
import com.sun.jdi.BooleanType
import org.apache.arrow.vector.types.FloatingPointPrecision
import org.apache.arrow.vector.types.pojo.ArrowType
import java.sql.SQLException

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
    name: String,
    op: String,
    l: LogicalExpr,
    r: LogicalExpr
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
    name: String,
    op: String,
    l: LogicalExpr,
    r: LogicalExpr
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
    val name: String,
    val expr: LogicalExpr
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
    val path: String,
    val dataSource: DataSource,
    val projection: List<String>
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




fun main() {

}