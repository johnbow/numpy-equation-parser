from core import *


name_count = 0
name_prefix = "f"

def standard_name():
    """Returns a generic expression name composed of the name_prefix and name_count."""
    global name_count
    name = name_prefix + str(name_count)
    name_count += 1
    return name


try:
    from varname import varname, VarnameRetrievingError
except (ImportError, VarnameRetrievingError):
    import sys
    print("Install 'varname' for better handling of compound functions.", file=sys.stderr)
    varname = standard_name


class ParserError(Exception):
    pass


class EqParser:

    class Element:
        BRACKET = 0
        STRING = 1
        NUMBER = 2
        OPERATOR = 3
        SEPARATOR = 4

        def __init__(self, identifier, content, level):
            self.identifier = identifier
            self.content = content
            self.level = level
            self.prec = -1
            self.ref = ""

        def add(self, char):
            self.content += char

    class Bracket(Element):
        def __init__(self, form, level, opening):
            super().__init__(EqParser.Element.BRACKET, form, level)
            self.is_opening = opening

    class UnidentifiedString(Element):
        def __init__(self, char, level):
            super().__init__(EqParser.Element.STRING, char, level)

    class UnidentifiedNumber(Element):
        def __init__(self, digit, level):
            super().__init__(EqParser.Element.NUMBER, digit, level)

    class UnidentifiedOperator(Element):
        def __init__(self, content, level, ref=""):
            super().__init__(EqParser.Element.OPERATOR, content, level)
            self.ref = ref

        @property
        def prec(self):
            return operators[self.content][PREC] + self.level * BRACKET_PREC

        @prec.setter
        def prec(self, value):
            pass

    class Separator(Element):
        def __init__(self):
            super().__init__(EqParser.Element.SEPARATOR, "sep", -1)

    # start of EqParser class

    def __init__(self, **options):
        self.options = {"do_cleanup": True,
                        "decimal_points": ",.",
                        "valid_opening_brackets": "([{",
                        "valid_closing_brackets": ")]}",
                        "merge_operators": ";|"}
        self.options.update(options)
        self.expressions = {}
        self.params: Dict[str, "Expression"] = {
            "e": Number(np.e),
            "pi": Number(np.pi),
            "inf": Number(np.inf),
            "i": Number(np.complex(0, 1))
        }

    def __getitem__(self, item):
        if item in self.expressions:
            return self.expressions[item]
        else:
            return self.params[item]

    def set_param(self, name: str, value: Union[str, complex, list, np.ndarray, "Expression"], var: dict = None):
        self.params[name] = np.nan
        if isinstance(value, Expression):
            self.params[name] = value
        elif isinstance(value, str):
            return self.parse(value, name, var=var)
        else:
            self.params[name] = as_expression(value, var, self.params)

    def parse(self, eqstr: str, name: str = "", var: dict = None) -> "Expression":
        """Parses the eqstr into an Expression and registers it in the parser."""
        if not name:
            try:
                name = varname()
            except VarnameRetrievingError:
                name = standard_name()
        var = var if var else {}
        flags = self._prepare_flags(eqstr)
        prec_list = self._list_precs(flags)
        root = self._parse_node(flags, prec_list, 0, len(flags), var)
        root.eqname = name
        self.expressions[name] = root
        return root

    def _prepare_flags(self, eqstr: str):
        """Returns a list of Expression Elements called flags which simplify parsing."""
        flags = self._fill_flags_unidentified(eqstr)
        if len(flags) == 0:
            raise ParserError("Could not identify any ExpressionElements.")
        self._identify_operators(flags)
        self._replace_unary_minus(flags)
        self._replace_pow10(flags)
        self._insert_multiply(flags)
        self._remove_brackets(flags)
        return flags

    def _remove_brackets(self, flags):
        i = 0
        while i < len(flags):
            if flags[i].identifier == self.Element.BRACKET:
                del flags[i]
            else:
                i += 1

    def _insert_multiply(self, flags):
        i = 1
        while i < len(flags):
            flag1, flag2 = flags[i-1:i+1]
            if flag1.identifier == self.Element.NUMBER:
                if flag2.identifier == self.Element.OPERATOR and operators[flag2.content][ARITY] == 1:
                    flags.insert(i, self.UnidentifiedOperator("*", flag1.level))
                elif flag2.identifier == self.Element.STRING:
                    flags.insert(i, self.UnidentifiedOperator("*", flag1.level))
                elif flag2.identifier == self.Element.BRACKET and flag2.is_opening:
                    flags.insert(i, self.UnidentifiedOperator("*", flag1.level))
            i += 1

    def _replace_pow10(self, flags):
        i = 1
        while i < len(flags):
            if flags[i].identifier == self.Element.OPERATOR and flags[i].content == "E":
                level = flags[i].level
                factor = float(flags[i-1].content)
                if flags[i+1].content == "-" or flags[i+1].content == "neg":
                    exponent = -float(flags[i+2].content)
                    del flags[i-1:i+3]
                else:
                    exponent = float(flags[i+1].content)
                    del flags[i-1:i+2]
                flags.insert(i-1, self.UnidentifiedNumber(str(pow10(factor, exponent)), level))
            else:
                i += 1

    def _identify_operators(self, flags):
        for i, flag in enumerate(flags):
            if flag.identifier != self.Element.STRING:
                continue
            if flag.content in self.expressions:
                flags[i] = self.UnidentifiedOperator("call", flag.level, ref=flag.content)
            elif flag.content in operators:
                flags[i] = self.UnidentifiedOperator(flag.content, flag.level)

    def _replace_unary_minus(self, flags):
        if flags[0].identifier == self.Element.OPERATOR and flags[0].content == "-":
            flags[0].content = "neg"
        for i, flag in enumerate(flags[1:]):
            if flag.identifier == self.Element.OPERATOR and flag.content == "-":
                if flags[i].identifier == self.Element.BRACKET and flags[i].is_opening:
                    flag.content = "neg"
                elif flags[i].identifier == self.Element.OPERATOR:
                    flag.content = "neg"
                else:
                    print(flags[i].content)

    def _fill_flags_unidentified(self, eqstr: str) -> List[Element]:
        """Flags every letter in eqstr as number, operator, etc."""
        flags = []
        level = 0
        for i, letter in enumerate(eqstr):
            if letter in self.options["valid_opening_brackets"]:
                level += 1
                flags.append(self.Bracket(letter, level, opening=True))
            elif letter in self.options["valid_closing_brackets"]:
                flags.append(self.Bracket(letter, level, opening=False))
                level -= 1
            elif letter.isalpha():
                if flags and flags[-1].identifier == self.Element.STRING:
                    flags[-1].add(letter)
                else:
                    flags.append(self.UnidentifiedString(letter, level))
            elif letter.isdigit() or letter in self.options["decimal_points"]:
                if flags and flags[-1].identifier == self.Element.NUMBER:
                    flags[-1].add(letter)
                else:
                    flags.append(self.UnidentifiedNumber(letter, level))
            elif letter.isspace() and flags and flags[-1].identifier != self.Element.SEPARATOR:
                flags.append(self.Separator())
            else:
                # might be +-/ etc. or just something different
                flags.append(self.UnidentifiedOperator(letter, level))
        i = 0
        while i < len(flags):   # now remove all separators
            if flags[i].identifier == self.Element.SEPARATOR:
                del flags[i]
            else:
                i += 1
        return flags

    def _list_precs(self, flags) -> List[int]:
        """Returns a list of flag indices sorted after highest precedence, then lowest index"""
        prec_list = [(i, flag.prec) for i, flag in enumerate(flags) if flag.prec >= 0]
        prec_list.sort(key=lambda item: (item[1], -item[0]))
        return [item[0] for item in prec_list]

    def _parse_node(self,
                    flags: List[Element],
                    prec_list: List[int],
                    left: int,
                    right: int,
                    var: dict) -> "Expression":
        """Parses nodes between left and right recursively and returns them in Expression-Form."""
        # find lowest precedence operator between left and right
        i = next(filter(lambda o: left <= o < right, prec_list), -1)
        if i < 0:
            if right - left > 1:
                raise ParserError(f"Encountered too many neighboring Expression Elements: {flags[left:right]}")
            return self._parse_primitive(left, flags, var)
        else:
            return self._parse_operator(i, flags, prec_list, left, right, var)

    def _parse_primitive(self, i, flags, var):
        if flags[i].identifier == self.Element.NUMBER:
            return Number(float(flags[i].content), var=var, params=self.params)
        if flags[i].identifier != self.Element.STRING:
            raise ParserError("Unexpected Expression Element: " + flags[i].content)
        content = flags[i].content
        if content in self.expressions:
            # merge var dicts
            var.update(self.expressions[content].var)
            self.expressions[content].change_vars(var)
            return Variable(content, var=var, params=self.params, eqref=self.expressions[content])
        elif content in self.params:
            return Parameter(content, var=var, params=self.params)
        else:
            var[content] = Number(np.nan)
            return Variable(content, var=var, params=self.params)

    def _parse_operator(self, i, flags, prec_list, left, right, var):
        opstr = flags[i].content
        if opstr in self.options["merge_operators"]:
            return self._parse_vector_operator(i, flags, prec_list, left, right, var)
        prec_list.remove(i)
        if opstr == "call":
            return self._parse_call_operator(i, flags, prec_list, left, right, var)
        else:
            return self._parse_standard_operator(i, flags, prec_list, left, right, var)

    def _parse_call_operator(self, i, flags, prec_list, left, right, var):
        right_op = self._parse_node(flags, prec_list, i+1, right, var)
        return NodeOperator("call", right_op, var=var, params=self.params,
                            expression=self.expressions[flags[i].ref])

    def _parse_standard_operator(self, i, flags, prec_list, left, right, var):
        opstr = flags[i].content
        number_of_nodes = operators[opstr][ARITY]
        if number_of_nodes == 1:
            right_op = self._parse_node(flags, prec_list, i+1, right, var)
            if opstr in node_operators:
                return NodeOperator(opstr, right_op, var=var, params=self.params)
            return Operator(opstr, right_op, var=var, params=self.params)
        elif number_of_nodes == 2:
            left_op = self._parse_node(flags, prec_list, left, i, var)
            right_op = self._parse_node(flags, prec_list, i+1, right, var)
            if opstr in node_operators:
                return NodeOperator(opstr, left_op, right_op, var=var, params=self.params)
            return Operator(opstr, left_op, right_op, var=var, params=self.params)
        else:
            raise ParserError(f"Operators can only have 1 or 2 arguments. Given: {number_of_nodes}")

    def _parse_vector_operator(self, i, flags, prec_list, left, right, var):
        vector_items = []
        common_level = flags[i].level
        index = left
        while index < right+1:
            if index >= right:
                vector_items.append(self._parse_node(flags, prec_list, left, index, var))
            elif flags[index].content in self.options["merge_operators"] and flags[index].level == common_level:
                prec_list.remove(index)
                vector_items.append(self._parse_node(flags, prec_list, left, index, var))
                left = index + 1
            index += 1
        return merge_expressions(vector_items, var=var, params=self.params)
