import re
from typing import Union, List, Callable, Tuple, Dict, Optional, Any

import numpy as np


max_repitions = 1_000_000


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
    if eqlist.args[0].name != "=":
        return np.sum(arg.value for arg in eqlist.args)
    if len(eqlist.args) != 3:
        raise ValueError(f"sum function expects 3 arguments. {len(eqlist.args)} given.")
    values = []
    k, upper, e = eqlist.args
    k = k.value
    i = 0
    while k.value <= upper.value:
        values.append(e.value)
        k += 1
        i += 1
        if i > max_repitions:
            print("Exceeded max repitions.")
            break
    return np.sum(values, axis=0)


def prod_function(eqlist: "Vector") -> Any:
    if eqlist.is_primitive:
        return np.prod(eqlist.value)
    if eqlist.args[0].name != "=":
        return np.prod(arg.value for arg in eqlist.args)
    if len(eqlist.args) != 3:
        raise ValueError(f"prod function expects 3 arguments. {len(eqlist.args)} given.")
    values = []
    k, upper, e = eqlist.args
    k = k.value
    i = 0
    while k.value <= upper.value:
        values.append(e.value)
        k += 1
        i += 1
        if i > max_repitions:
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


ARGS, PREC, FUNC = 0, 1, 2
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
    # vector functions
    "cross": (2, 500, np.cross),
    "dot": (2, 500, np.dot),
    # special functions
    "max": (1, 600, max),
    "min": (1, 600, min),
}

node_operators: Dict[str, Tuple[int, int, Callable]] = {
    "=": (2, 2, assign),
    "sum": (1, 600, sum_function),
    "prod": (1, 600, prod_function),
}
operators.update(node_operators)


def add_operator(opname: str, func: Callable, args: int, prec: int = -1, node_operator: bool = False):
    """
    Adds 'opname' to the list of operators.
    """
    if opname in operators:
        raise ValueError(f"Operator '{opname}' already exists!")
    elif not opname.isalpha():
        raise ValueError(f"Operator name should only contain letters.")
    if prec < 0:
        prec = 500 + 100 * int(args == 2)
    operators[opname] = (args, prec, func)
    if node_operator:
        node_operators[opname] = (args, prec, func)


def _norm_eq(eqstr: str) -> str:
    eqstr = eqstr.replace(",", ".").replace("_", "")
    i = 0
    while i < len(eqstr) - 1:
        this, nxt = eqstr[i:i + 2]
        if this.isdigit() and nxt.isalpha():
            eqstr = eqstr[:i + 1] + "*" + eqstr[i + 1:]
        i += 1
    return eqstr


def _list_precs(eqstr: str) -> List[Tuple[str, int, int]]:
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
            if opstr not in operators:
                opstr = ""
                continue
            oplist = operators[opstr]
            prec = oplist[PREC] + level * BRACKET_PREC
            index = i - len(opstr) + 1
            precs.append((opstr, index, prec))
            opstr = ""

    return precs


class EqParser:
    standard_options = {
        "std_name": "f",
        "init_name_count": 0
    }

    def __init__(self, options: dict = None):
        self.options = options if options else {}
        self.options = dict(EqParser.standard_options, **self.options)
        self.equations = {}
        self.params: Dict[str, "Equation"] = {
            "e": Number(np.e),
            "pi": Number(np.pi),
            "inf": Number(np.inf)
        }
        self.name_counter = self.options["init_name_count"]

    def set_param(self, name: str, value: Union[str, float, list, np.ndarray, "Equation"], var: dict = None):
        self.params[name] = np.nan
        if isinstance(value, Equation):
            self.params[name] = value
        else:
            self.params[name] = self.as_equation(value, name, var)

    def as_equation(self, value: Union[str, float, list, np.ndarray], name: str = "", var: dict = None) -> "Equation":
        if isinstance(value, float):
            return Number(value, var=var, params=self.params)
        elif isinstance(value, list):
            return Vector(np.array(value), var=var, params=self.params)
        elif isinstance(value, np.ndarray):
            return Vector(value, var=var, params=self.params)
        elif isinstance(value, str):
            return self.parse(value, name, var=var)
        raise ValueError(f"Type {str(type(value))} can not be converted to an equation.")

    def parse(self, eqstr: str, name: str = "", var: dict = None) -> "Equation":
        name = name if name else self.standard_name()
        var = var if var else {}
        eqstr = _norm_eq(eqstr)
        prec_list = _list_precs(eqstr)
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
        op_list = next(filter(lambda op: left <= op[1] < right, prec_list), None)
        if not op_list:
            # this is an end node
            content = re.sub(r"[()\[\] ]", "", eqstr[left:right])
            if not content:
                content = "0"
            if content.replace('.', '', 1).isdigit():
                return Number(float(content), var=var, params=self.params)
            if content in self.params:
                return Parameter(content, var=var, params=self.params)
            else:
                var[content] = np.nan
                return Variable(content, var=var, params=self.params)
        else:
            # this node is an operator
            prec_list.remove(op_list)
            opstr, index, prec = op_list
            if opstr == ";" or opstr == "|":  # check for Vector operators
                vector_ops = list(filter(lambda op: left <= op[1] < right and prec == op[2], prec_list))
                for op in vector_ops:
                    prec_list.remove(op)
                vector_ops = [op_list] + vector_ops
                indices: List[Union[None, int]] = [left-1] + [op[1] for op in reversed(vector_ops)] + [right]
                vector_items = [self._parse_node(eqstr, prec_list, i+1, j, var)
                                for i, j in zip(indices, indices[1:])]
                return merge_equations(vector_items, var=var, params=self.params)
            num_args = operators[opstr][ARGS]
            if num_args == 1:
                right = self._parse_node(eqstr, prec_list, index+len(opstr), right, var)
                if opstr in node_operators:
                    return NodeOperator(opstr, right, var=var, params=self.params)
                return Operator(opstr, right, var=var, params=self.params)
            elif num_args == 2:
                left = self._parse_node(eqstr, prec_list, left, index, var)
                right = self._parse_node(eqstr, prec_list, index+len(opstr), right, var)
                if opstr in node_operators:
                    return NodeOperator(opstr, left, right, var=var, params=self.params)
                return Operator(opstr, left, right, var=var, params=self.params)
            else:
                raise ValueError("Only operators with 1 or 2 arguments are allowed.")

    def standard_name(self) -> str:
        name = self.options["std_name"] + str(self.name_counter)
        self.name_counter += 1
        return name


class Equation:

    def __init__(self,
                 *args: "Equation",
                 eqstr: str = "",
                 name: str = "",
                 var: dict = None,
                 params: dict = None,
                 value: Union[float, np.ndarray] = np.nan):
        self.args = args
        self.name = name
        self._value = value
        self.origstr = eqstr
        self.var = var
        self.params = params
        self.eqname = ""
        self.is_primitive = False   # True if value accesses own node's memory directly
        self.is_assignable = False  # True if value accesses another node's value
        self.__iter__ = self.args.__iter__
        self.__contains__ = self.args.__contains__
        self.__len__ = self.args.__len__

    @property
    def value(self) -> Union[float, np.ndarray]:
        return self._value

    @value.setter
    def value(self, v: Union[float, np.ndarray]):
        self._value = v

    def set_var(self, name: str, value: Union["Equation", float, np.ndarray]):
        if isinstance(value, Equation):
            self.var[name] = value

    def __call__(self, *args: Union[float, np.ndarray], **kwargs) -> Union[float, np.ndarray]:
        if args:
            for k, v in zip(sorted(self.var.keys()), args):
                self.var[k] = v
        if kwargs:
            self.var.update(kwargs)
        return self.value

    def __iadd__(self, other):
        self.value += other
        return self

    def __getitem__(self, item):
        # use for slicing
        pass

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Equation(name={self.name}, vars={str(self.var)}, params={str(self.params)})"


class Operator(Equation):

    def __init__(self, opstr: str, *args, var: dict = None, params: dict = None):
        super().__init__(*args, name=opstr, var=var, params=params)
        self.opstr = opstr
        self.func = operators[opstr][FUNC]

    @property
    def value(self) -> Union[float, np.ndarray]:
        return self.func(*[item.value for item in self.args])

    def __str__(self):
        if len(self.args) == 1:
            return f"{self.opstr}( {str(self.args[0])} )"
        else:
            left = str(self.args[0])
            if type(self.args[0]) == Operator and \
                    operators[self.args[0].opstr][PREC] < operators[self.opstr][PREC]:
                left = "(" + left + ")"
            right = str(self.args[1])
            if type(self.args[1]) == Operator and \
                    operators[self.args[1].opstr][PREC] < operators[self.opstr][PREC]:
                right = "(" + right + ")"
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
    def value(self) -> Union[float, np.ndarray, "Equation"]:
        return self.func(*self.args)


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

    def __repr__(self):
        return f"Operator(name='{self.name}')"


class Variable(Equation):

    def __init__(self, name: str, var: dict, params: dict = None):
        super().__init__(name=name, var=var, params=params)
        self.is_assignable = True

    @property
    def value(self) -> Union[float, np.ndarray]:
        return self.var[self.name].value

    @value.setter
    def value(self, v: Union[float, np.ndarray, "Equation"]):
        self.set_var(self.name, v)

    def __repr__(self):
        return f"Variable(name='{self.name}', value={self.value})"


class Number(Equation):

    def __init__(self, value: float, var: dict = None, params: dict = None):
        super().__init__(name=str(value), value=value, var=var, params=params)
        self.is_primitive = True

    def __repr__(self):
        return f"Number(value={self.value})"


class Vector(Equation):

    def __init__(self, *args: Union[np.ndarray, "Equation"], var: dict = None, params: dict = None):
        super().__init__(*args, var=var, params=params)
        self.is_primitive = len(self.args) == 1
        if self.is_primitive:
            self._value = lambda: self.args[0]
        else:
            self._value = lambda: np.array([arg.value for arg in self.args])

    @property
    def value(self) -> np.ndarray:
        return self._value()

    @value.setter
    def value(self, v: np.ndarray):
        self.args = tuple(v)
        self.is_primitive = True

    def __str__(self):
        if self.is_primitive:
            return f"[{'; '.join(map(str, self.args[0]))}]"
        else:
            return f"[{'; '.join(map(str, self.args))}]"

    def __repr__(self):
        return f"EquationList({'; '.join(map(repr, self.args))})"


if __name__ == "__main__":
    parser = EqParser()
    f = parser.parse("sum(x=1;inf; 1/x^2)")
    print(f())