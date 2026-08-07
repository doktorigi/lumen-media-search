from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from std.sys import simd_width_of


@export
def PyInit_mojo_ranker() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("mojo_ranker")
        module.def_function[top_k](
            "top_k",
            docstring="Exact top-k dot products for normalized float32 NumPy arrays.",
        )
        return module.finalize()
    except error:
        abort(String("could not initialize mojo_ranker: ", error))


def top_k(
    matrix_object: PythonObject,
    query_object: PythonObject,
    k_object: PythonObject,
) raises -> PythonObject:
    """Return deterministic top-k indices and scores without copying array data."""
    var rows = Int(py=matrix_object.shape[0])
    var dimensions = Int(py=matrix_object.shape[1])
    var requested = Int(py=k_object)
    var count = min(max(requested, 0), rows)

    var output_indices = Python.list()
    var output_scores = Python.list()
    if count == 0:
        return Python.tuple(output_indices, output_scores)

    var matrix = matrix_object.ctypes.data.unsafe_get_as_pointer[DType.float32]()
    var query = query_object.ctypes.data.unsafe_get_as_pointer[DType.float32]()
    var best_indices = List[Int](capacity=count)
    var best_scores = List[Float32](capacity=count)
    for _ in range(count):
        best_indices.append(-1)
        best_scores.append(-1.0e30)

    comptime width = simd_width_of[DType.float32]()
    for row in range(rows):
        var vector_sum = SIMD[DType.float32, width](0)
        var column = 0
        var base = row * dimensions
        while column + width <= dimensions:
            vector_sum += (
                matrix.load[width=width](base + column)
                * query.load[width=width](column)
            )
            column += width

        var score = Float32(vector_sum.reduce_add()[0])
        while column < dimensions:
            score += matrix[base + column] * query[column]
            column += 1

        var insertion = count
        for position in range(count):
            if score > best_scores[position]:
                insertion = position
                break
        if insertion < count:
            for position in range(count - 1, insertion, -1):
                best_scores[position] = best_scores[position - 1]
                best_indices[position] = best_indices[position - 1]
            best_scores[insertion] = score
            best_indices[insertion] = row

    for position in range(count):
        output_indices.append(best_indices[position])
        output_scores.append(best_scores[position])
    return Python.tuple(output_indices, output_scores)
