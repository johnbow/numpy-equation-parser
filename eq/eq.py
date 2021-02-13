import re
from typing import Union, List, Callable, Tuple, Dict, Optional, Any

import numpy as np

np.seterr(all="ignore")

name_count = 0
name_prefix = "f"

def standard_name():
    global name_count
    name = name_prefix + str(name_count)
    name_count += 1
    return name


try:
    from varname import varname
except ImportError:
    import sys
    print("Install 'varname' for better handling of compound functions.", file=sys.stderr)
    varname = standard_name


MAX_REPITIONS = 1_000_000


def kth_root(k: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    if k % 2 != 0:
        res = np.power(np.abs(x), 1. / k)
        return res * np.sign(x)
    else:
        return np.power(np.abs(x), 1. / k)

def logbase(base: float, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return np.log(x) / np.log(base)

def sum_function(eqlist: "Vector") -> Any:
    if eqlist.is_primitive:
        return np.sum(eqlist.value)
    if eqlist.nodes[0].name != "=":
        return np.sum(node.value for node in eqlist.nodes)
    if len(eqlist.nodes) != 3:
        raise ValueError(f"sum function expects 3 arguments. {len(eqlist.nodes)} given.")
    values = []
    k, upper, e = eqlist.nodes
    k = k.value
    i = 0
    while k.value <= upper.value:
        values.append(e.value)
        k += 1
        i += 1
        if i > MAX_REPITIONS:
            print("Exceeded max repitions.")
            break
    return np.sum(values, axis=0)


def prod_function(eqlist: "Vector") -> Any:
    if eqlist.is_primitive:
        return np.prod(eqlist.value)
    if eqlist.nodes[0].name != "=":
        return np.prod(node.value for node in eqlist.nodes)
    if len(eqlist.nodes) != 3:
        raise ValueError(f"prod function expects 3 arguments. {len(eqlist.nodes)} given.")
    values = []
    k, upper, e = eqlist.nodes
    k = k.value
    i = 0
    while k.value <= upper.value:
        values.append(e.value)
        k += 1
        i += 1
        if i > MAX_REPITIONS:
            print("Exceeded max repitions.")
            break
    return np.prod(values, axis=0)


def merge_equations(eqs: List["Equation"], var: dict = None, params: dict = None) -> "Vector":
    if all(e.is_primitive for e in eqs):
        return Vector(np.array([e.value for e in eqs]), var=var, params=params)
    else:
        return Vector(*eqs, var=var, params=params)

def assign(node1: "Equation", node2: "Equation") -> Optional["Equation"]:
    """
    Assigns one node to the other and returns the node to which the equation or value has been assigned to.

    If there is no node which can be assigned to, the function returns None.
    If both nodes are variables or parameters, the right node will be assigned to the left.
    """
    if node1.is_assignable and node2.is_assignable:
        left = node1
        right = node2
    elif node1.is_assignable or node2.is_assignable:
        left = node1 if node1.is_assignable else node2
        right = node1 if node2.is_assignable else node2
    else:
        return None
    left.value = right.value
    return left

def call_equation(inner: "Equation", equation: "Equation") -> Union[float, np.ndarray, "Equation"]:
    variables = []

    def set_vars(_inner: "Equation"):
        inner_value = _inner.value
        if isinstance(inner_value, Variable):
            equation.set_var(inner_value.name, inner_value.value)
        elif isinstance(_inner, Vector) and not _inner.is_primitive:
            for node in _inner.nodes:
                set_vars(node)
        else:
            variables.append(inner)

    if isinstance(inner, Equation):
        set_vars(inner)
    else:
        variables.append(inner)
    equation.set_vars(*variables)
    return equation.value


NODES, PREC, FUNC = 0, 1, 2
BRACKET_PREC = 1000
# abbrev, args, prec, func
operators: Dict[str, Tuple[int, int, Callable]] = {
    # parser operators
    "|": (2, 1, merge_equations),
    ";": (2, 1, merge_equations),
    # logic operators
    "or": (2, 40, np.logical_or),
    "and": (2, 50, np.logical_and),
    "xor": (2, 60, np.logical_xor),
    "not": (2, 70, np.logical_not),
    # basic operators
    "+": (2, 200, np.add),
    "-": (2, 200, np.subtract),
    "*": (2, 300, np.multiply),
    "/": (2, 300, np.divide),
    ":": (2, 300, np.divide),
    "^": (2, 400, np.power),
    # basic functions
    "root": (2, 500, kth_root),
    "mod": (2, 500, np.mod),
    "abs": (1, 600, np.abs),
    "neg": (1, 600, np.negative),
    "sqrt": (1, 600, np.sqrt),
    "exp": (1, 600, np.exp),
    "ln": (1, 600, np.log),
    "log": (2, 500, logbase),
    # trigonometric functions
    "sin": (1, 600, np.sin),
    "cos": (1, 600, np.cos),
    "tan": (1, 600, np.tan),
    "arcsin": (1, 600, np.arcsin),
    "arccos": (1, 600, np.arccos),
    "arctan": (1, 600, np.arctan),
    "degrees": (1, 600, np.degrees),
    "radians": (1, 600, np.radians),
    # hyperbolic functions
    "sinh": (1, 600, np.sinh),
    "cosh": (1, 600, np.cosh),
    "tanh": (1, 600, np.tanh),
    "arcsinh": (1, 600, np.arcsinh),
    "arccosh": (1, 600, np.arccosh),
    "arctanh": (1, 600, np.arctanh),
    # round functions
    "round": (1, 600, np.rint),
    "floor": (1, 600, np.floor),
    "ceil": (1, 600, np.ceil),
    "trunc": (1, 600, np.trunc),
    # vector functions
    "cross": (2, 500, np.cross),
    "dot": (2, 500, np.dot),
    "max": (1, 600, max),
    "min": (1, 600, min),
}

node_operators: Dict[str, Tuple[int, int, Callable]] = {
    "=": (2, 2, assign),
    "call": (1, 600, call_equation),
    "sum": (1, 600, sum_function),
    "prod": (1, 600, prod_function),
}
operators.update(node_operators)


def add_operator(opname: str, func: Callable, nodes: int, prec: int = -1, node_operator: bool = False):
    """
    Adds 'opname' to the list of operators.
    """
    if opname in operators:
        raise ValueError(f"Operator '{opname}' already exists!")
    elif not opname.isalpha():
        raise ValueError(f"Operator name should only contain letters.")
    if prec < 0:
        prec = 500 + 100 * int(nodes == 2)
    operators[opname] = (nodes, prec, func)
    if node_operator:
        node_operators[opname] = (nodes, prec, func)


def _norm_eq(eqstr: str) -> str:
    eqstr = eqstr.replace(",", ".").replace("_", "")
    i = 0
    while i < len(eqstr) - 1:
        this, nxt = eqstr[i:i + 2]
        if (this.isdigit() and nxt.isalpha()) or (this.isdigit() and nxt == "("):
            eqstr = eqstr[:i + 1] + "*" + eqstr[i + 1:]
        i += 1
    return eqstr


def _list_precs(eqstr: str, equations: dict = None) -> List[Tuple[str, int, int]]:
    equations = equations if equations else {}
    precs = []  # (opstr, index, prec)
    level = 0
    opstr = ""  # can also be variable or parameter
    for i, letter in enumerate(eqstr):
        next_letter = eqstr[i + 1] if i + 1 < len(eqstr) else ""
        if letter == "(" or letter == "[":
            level += 1
        elif letter == ")" or letter == "]":
            level -= 1
        elif letter.isalpha():
            opstr += letter
        elif not letter.isdigit():  # letter is prob +-/*:
            opstr += letter
        # if opstr is operator
        if opstr and (not next_letter or not next_letter.isalpha() or not opstr.isalpha()):
            if opstr not in operators and opstr not in equations:
                opstr = ""
                continue
            elif opstr in equations:
                if next_letter == "(":
                    prec = 600 + level * BRACKET_PREC
                    index = i - len(opstr) + 1
                    precs.append((opstr, index, prec))
            else:
                oplist = operators[opstr]
                prec = oplist[PREC] + level * BRACKET_PREC
                index = i - len(opstr) + 1
                precs.append((opstr, index, prec))
            opstr = ""
    return precs


def as_equation(value: Union[complex, list, np.ndarray], var: dict = None, params: dict = None) -> "Equation":
    if isinstance(value, (float, int, complex)):
        return Number(value, var=var, params=params)
    elif isinstance(value, np.ndarray):
        return Vector(value, var=var, params=params)
    elif isinstance(value, list):
        return Vector(np.array(value), var=var, params=params)
    raise ValueError(f"Type {str(type(value))} can not be converted to an equation.")


class EqParser:
    standard_options = {}

    def __init__(self, options: dict = None):
        self.options = options if options else {}
        self.options = dict(EqParser.standard_options, **self.options)
        self.equations = {}
        self.params: Dict[str, "Equation"] = {
            "e": Number(np.e),
            "pi": Number(np.pi),
            "inf": Number(np.inf),
            "i": Number(np.complex(0, 1))
        }

    def set_param(self, name: str, value: Union[str, complex, list, np.ndarray, "Equation"], var: dict = None):
        self.params[name] = np.nan
        if isinstance(value, Equation):
            self.params[name] = value
        elif isinstance(value, str):
            return self.parse(value, name, var=var)
        else:
            self.params[name] = as_equation(value, var, self.params)

    def parse(self, eqstr: str, name: str = "", var: dict = None) -> "Equation":
        if not name:
            name = varname()
        while not name or name in self.equations:
            name = standard_name()
        var = var if var else {}
        eqstr = _norm_eq(eqstr)
        prec_list = _list_precs(eqstr, self.equations)
        # sort after precedence, then index
        prec_list.sort(key=lambda item: (item[2], -item[1]))
        root = self._parse_node(eqstr, prec_list, 0, len(eqstr), var)
        root.eqname = name
        self.equations[name] = root
        return root

    def _parse_node(self,
                    eqstr: str,
                    prec_list: List[Tuple[str, int, int]],
                    left: int,
                    right: int,
                    var: dict) -> "Equation":
        # find lowest precedence operator between left and right
        op_list = next(filter(lambda o: left <= o[1] < right, prec_list), None)
        if not op_list:
            # this is an end node
            content = re.sub(r"[()\[\] ]", "", eqstr[left:right])
            if not content:
                content = "0"
            if content.replace('.', '', 1).isdigit():
                return Number(float(content), var=var, params=self.params)

            if content in self.equations:
                # merge var dicts
                var.update(self.equations[content].var)
                self.equations[content].change_vars(var)
                return Variable(content, var=var, params=self.params, eqref=self.equations[content])
            elif content in self.params:
                return Parameter(content, var=var, params=self.params)
            else:
                var[content] = Number(np.nan)
                return Variable(content, var=var, params=self.params)
        else:
            # this node is an operator
            prec_list.remove(op_list)
            opstr, index, prec = op_list
            if opstr == ";" or opstr == "|":  # check for Vector operators
                vector_ops = list(filter(lambda o: left <= o[1] < right and prec == o[2], prec_list))
                for op in vector_ops:
                    prec_list.remove(op)
                vector_ops = [op_list] + vector_ops
                indices: List[Union[None, int]] = [left-1] + [op[1] for op in reversed(vector_ops)] + [right]
                vector_items = [self._parse_node(eqstr, prec_list, i+1, j, var)
                                for i, j in zip(indices, indices[1:])]
                return merge_equations(vector_items, var=var, params=self.params)

            if opstr in self.equations:
                right_op = self._parse_node(eqstr, prec_list, index + len(opstr), right, var)
                return NodeOperator("call", right_op, var=var, params=self.params, equation=self.equations[opstr])

            number_of_nodes = operators[opstr][NODES]
            if number_of_nodes == 1:
                right_op = self._parse_node(eqstr, prec_list, index+len(opstr), right, var)
                if opstr in node_operators:
                    return NodeOperator(opstr, right_op, var=var, params=self.params)
                return Operator(opstr, right_op, var=var, params=self.params)
            elif number_of_nodes == 2:
                left_op = self._parse_node(eqstr, prec_list, left, index, var)
                right_op = self._parse_node(eqstr, prec_list, index+len(opstr), right, var)
                if opstr in node_operators:
                    return NodeOperator(opstr, left_op, right_op, var=var, params=self.params)
                return Operator(opstr, left_op, right_op, var=var, params=self.params)
            else:
                raise ValueError("Only operators with 1 or 2 arguments are allowed.")

    def __getitem__(self, item):
        if item in self.equations:
            return self.equations[item]
        else:
            return self.params[item]


class Equation:

    def __init__(self,
                 *nodes: "Equation",
                 eqstr: str = "",
                 name: str = "",
                 var: dict = None,
                 params: dict = None,
                 value: Union[complex, np.ndarray] = np.nan,
                 **kwargs):
        self.nodes = nodes
        self.kwargs = kwargs
        self.name = name
        self._value = value
        self.origstr = eqstr
        self.var = var
        self.params = params
        self.eqname = ""
        self.opstr = ""
        self.is_primitive = False   # True if value accesses own node's memory directly
        self.is_assignable = False  # True if value accesses another node's value
        self.is_eqref = False       # True if value accesses contained equation as reference

    @property
    def value(self) -> Union[complex, np.ndarray]:
        return self._value

    @value.setter
    def value(self, v: Union[complex, np.ndarray]):
        self._value = v

    def change_vars(self, new_vars: dict):
        self.var = new_vars
        for node in self.nodes:
            node.change_vars(new_vars)

    def merge_vars(self, other: "Equation"):
        self.var.update(other.var)
        other.change_vars(self.var)

    def set_var(self, name: str, value: Union[complex, np.ndarray, "Equation"]) -> None:
        if isinstance(value, Equation):
            self.var[name] = value
        else:
            self.var[name] = as_equation(value, self.var, self.params)

    def set_vars(self, *args: Union[complex, np.ndarray, "Equation"], **kwargs):
        for k, v in zip(sorted(self.var.keys()), args):
            self.set_var(k, v)
        if kwargs:
            for k, v in kwargs.items():
                self.set_var(k, v)

    def __call__(self, *args: Union[complex, np.ndarray, "Equation"], **kwargs) -> Union[float, np.ndarray, "Equation"]:
        self.set_vars(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], Equation):
            return self
        return self.value

    def __getitem__(self, item: Union[tuple, slice]) -> np.ndarray:
        """
        Inputs an array into the function and returns the calculated value.

        The input can be either a slice or a tuple of values. If it is a slice,
        the array will be made using np.arange, if it is a list of values, the values
        are going to be wrapped into an array. Consider that this works only for
        equations with one variables. For more variables, each array should be passed
        to __call__.
        """
        if isinstance(item, tuple):
            return self.__call__(np.array(item))
        else:
            return self.__call__(np.arange(item.start, item.stop, item.step))

    def __iadd__(self, other: Union[float, np.ndarray]):
        self.value += other
        return self

    def __isub__(self, other: Union[float, np.ndarray]):
        self.value -= other
        return self

    def __imul__(self, other: Union[float, np.ndarray]):
        self.value *= other
        return self

    def __idiv__(self, other: Union[float, np.ndarray]):
        self.value /= other
        return self

    def __itruediv__(self, other: Union[float, np.ndarray]):
        self.value /= other
        return self

    def __ifloordiv__(self, other: Union[float, np.ndarray]):
        self.value //= other
        return self

    def __imod__(self, other: Union[float, np.ndarray]):
        self.value %= other
        return self

    def __ipow__(self, other: Union[float, np.ndarray]):
        self.value **= other
        return self

    def __iand__(self, other: Union[float, np.ndarray]):
        self.value &= other
        return self

    def __ior__(self, other: Union[float, np.ndarray]):
        self.value |= other
        return self

    def __ixor__(self, other: Union[float, np.ndarray]):
        self.value ^= other
        return self

    def __ilshift__(self, other: Union[float, np.ndarray]):
        self.value <<= other
        return self

    def __irshift__(self, other: Union[float, np.ndarray]):
        self.value >>= other
        return self

    def __add__(self, other: "Equation"):
        self.merge_vars(other)
        return Operator("+", self, other, var=self.var, params=self.params)

    def __sub__(self, other: "Equation"):
        self.merge_vars(other)
        return Operator("-", self, other, var=self.var, params=self.params)

    def __mul__(self, other: "Equation"):
        self.merge_vars(other)
        return Operator("*", self, other, var=self.var, params=self.params)

    def __truediv__(self, other: "Equation"):
        self.merge_vars(other)
        return Operator("/", self, other, var=self.var, params=self.params)

    def __floordiv__(self, other: "Equation"):
        self.merge_vars(other)
        op = Operator("/", self, other, var=self.var, params=self.params)
        return Operator("floor", op, var=self.var, params=self.params)

    def __mod__(self, other: "Equation"):
        self.merge_vars(other)
        return Operator("mod", self, other, var=self.var, params=self.params)

    def __pow__(self, power: "Equation", modulo=None):
        self.merge_vars(power)
        return Operator("^", self, power, var=self.var, params=self.params)

    def __or__(self, other: "Equation"):
        self.merge_vars(other)
        return Operator("or", self, other, var=self.var, params=self.params)

    def __xor__(self, other: "Equation"):
        self.merge_vars(other)
        return Operator("xor", self, other, var=self.var, params=self.params)

    def __and__(self, other: "Equation"):
        self.merge_vars(other)
        return Operator("and", self, other, var=self.var, params=self.params)

    def __floor__(self):
        return Operator("floor", self, var=self.var, params=self.params)

    def __ceil__(self):
        return Operator("ceil", self, var=self.var, params=self.params)

    def __trunc__(self):
        return Operator("trunc", self, var=self.var, params=self.params)

    def __abs__(self):
        return Operator("abs", self, var=self.var, params=self.params)

    def __pos__(self):
        return self

    def __neg__(self):
        return Operator("neg", self, var=self.var, params=self.params)

    def __invert__(self):
        self.nodes = self.nodes[::-1]

    def __round__(self):
        return Operator("round", self, var=self.var, params=self.params)

    def __lt__(self, other: "Equation"):
        return self.value < other.value

    def __le__(self, other: "Equation"):
        return self.value <= other.value

    def __eq__(self, other: "Equation"):
        return self.value == other.value

    def __ne__(self, other: "Equation"):
        return self.value != other.value

    def __ge__(self, other: "Equation"):
        return self.value >= other.value

    def __gt__(self, other: "Equation"):
        return self.value > other.value

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __complex__(self):
        return complex(self.value)

    def __hex__(self):
        return hex(self.value)

    def __iter__(self):
        return iter(self.nodes)

    def __copy__(self):
        pass

    def __deepcopy__(self, memodict=None):
        memodict = memodict if memodict is not None else {}

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Equation(name={self.name}, vars={str(self.var)}, params={str(self.params)})"


class Operator(Equation):

    def __init__(self, opstr: str, *nodes, var: dict = None, params: dict = None, **kwargs):
        super().__init__(*nodes, name=opstr, var=var, params=params, **kwargs)
        self.opstr = opstr
        self.func = operators[opstr][FUNC]

    @property
    def value(self) -> Union[complex, np.ndarray]:
        return self.func(*[item.value for item in self.nodes], **self.kwargs)

    def __str__(self):
        if self.opstr == "call":
            name = self.kwargs['equation'].eqname
            name = name if name else self.opstr
            return f"{name}({str(self.nodes[0])})"
        if len(self.nodes) == 1:
            return f"{self.opstr}({str(self.nodes[0])})"
        else:

            left_node = self.nodes[0] if not self.nodes[0].is_eqref else self.nodes[0]._value
            if self.opstr == "-" and isinstance(left_node, Number) and left_node.value == 0:
                left = ""   # unary operator: 0-x -> -x
            elif isinstance(left_node, Operator) and operators[left_node.opstr][PREC] < operators[self.opstr][PREC]:
                left = "(" + str(left_node) + ")"
            else:
                left = str(left_node)

            right_node = self.nodes[1] if not self.nodes[1].is_eqref else self.nodes[1]._value
            if isinstance(right_node, Operator) and operators[right_node.opstr][PREC] < operators[self.opstr][PREC]:
                right = "(" + str(right_node) + ")"
            else:
                right = str(right_node)

            return " ".join((left, self.opstr, right))

    def __repr__(self):
        return f"Operator(opstr='{self.opstr}', func={self.func})"


class NodeOperator(Operator):
    """
    Passes the child nodes instead of their values to the evaluating function.

    This can be used to enable a function to customize its return value to another type
    than float or array or to access single Vector elements directly.
    """

    @property
    def value(self) -> Union[complex, np.ndarray, "Equation"]:
        return self.func(*self.nodes, **self.kwargs)


class Parameter(Equation):

    def __init__(self, name, params: dict, var: dict = None):
        super().__init__(name=name, var=var, params=params)
        self.is_assignable = True

    @property
    def value(self) -> Union[float, np.ndarray]:
        return self.params[self.name].value

    @value.setter
    def value(self, v: Union[float, np.ndarray]):
        self.params[self.name].value = v

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Operator(name='{self.name}')"


class Variable(Equation):

    def __init__(self, name: str, var: dict, params: dict = None, eqref: "Equation" = None):
        super().__init__(name=name, var=var, params=params)
        self.is_assignable = True
        self._value = eqref
        self.is_eqref = eqref is not None

    @property
    def value(self) -> Union[float, np.ndarray]:
        if self._value is None:
            return self.var[self.name].value
        return self._value.value

    @value.setter
    def value(self, v: Union[float, np.ndarray, "Equation"]):
        self.set_var(self.name, v)

    def __str__(self):
        if self._value is None:
            return self.name
        return f"{str(self._value)}"

    def __repr__(self):
        return f"Variable(name='{self.name}', value={self.value})"


class Number(Equation):

    def __init__(self, value: complex, var: dict = None, params: dict = None):
        super().__init__(name=str(value), value=value, var=var, params=params)
        self.is_primitive = True

    def __repr__(self):
        return f"Number(value={self.value})"


class Vector(Equation):

    def __init__(self,
                 vec_or_item1: Union[np.ndarray, "Equation"],
                 *nodes: Union[np.ndarray, "Equation"],
                 var: dict = None,
                 params: dict = None):
        super().__init__(*nodes, var=var, params=params)
        self.is_primitive = len(nodes) == 0
        if self.is_primitive:
            self.vec = vec_or_item1
            self._value = lambda: self.vec
        else:
            self.nodes = (vec_or_item1,) + self.nodes
            self._value = lambda: np.array([node.value for node in self.nodes])

    @property
    def value(self) -> np.ndarray:
        return self._value()

    @value.setter
    def value(self, v: np.ndarray):
        self.vec = tuple(v)
        self.is_primitive = True

    def __str__(self):
        if self.is_primitive:
            return f"[{'; '.join(map(str, self.vec))}]"
        else:
            return f"[{'; '.join(map(str, self.nodes))}]"

    def __repr__(self):
        return f"Vector({'; '.join(map(repr, self.nodes))})"


if __name__ == "__main__":
    # import timeit
    # print(timeit.timeit(calc, number=100))
    parser = EqParser()
    f = parser.parse("2*3i")

# TODO: overwrite names after next assignment
# TODO: write README
# TODO: write docs
# TODO: implement visualizer in other file
