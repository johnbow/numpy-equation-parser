# numpy-equation-parser
Modules to quickly parse and use functions for graphing and computation.

## Installation
Just copy the eq module into your project. Then you can import it with `from eq.core import EqParser`.
For using the on-the-fly name parsing feature, you have to install
the [varname](https://github.com/pwwang/python-varname) module by 
typing `pip install -U varname`.

## Example
Here is an example code snippet that shows the module's functionality.

```java
parser = EqParser()
f = parser.parse("3x+4")
g = parser.parse("sin(t)")
	
str(f)
>>> 3.0 * x + 4.0

f(4)
>>> 16.0

g[1:5]
>>> [ 0.84147098  0.90929743  0.14112001 -0.7568025 ]
	
h = parser.parse("f(x=g)")
h(np.pi)
>>> 4.0
```

## Tutorial

### Parser
Instantiate the parser directly. All parameters and equations belong to the parser and can 
be retrieved by their names with the syntax `parser[name]`. Equations are stored in a tree structure 
for fast calculation purposes.
```python
from eq.core import EqParser

parser = eq.EqParser()
f = parser.parse("3 root 125")
parser["f"].value	# equivalent to f.value
>>> 5.0
```

_____________________________

### Variables and Parameters
Parameters are shared between all equations instantiated by the same parser and are
declared by the same. Following parameters can be used without the need of declaration: `e, pi, inf, i`.
```python
A = np.arange(6).reshape(2, 3)
parser.set_param("A", A)
f = parser.parse("3*A")
f
>>> [[ 0.  3.  6.]
     [ 9. 12. 15.]]

pi_param = parser.parse("pi")
type(pi_param)
>>> <class 'eq.Parameter'>
```

Variables are unique to every equation, therefore equations can declare independent variables 
that have the same name as other variables in other equations. By default, undefinded strings 
are parsed to variables instead of parameters.
```python
f = parser.parse("e^x")
f.var
>>> {'x': Number(value=nan)}

f.set_vars(x=2)
f.var
>>> {'x': Number(value=2)}

f()
>>> 7.3890560989306495
```
By calling the equation object the passed arguments set the variables accordingly. If no name 
is provided they are set in alphabetical order.
```python
f = parser.parse("sqrt(x^2 - y^2)")
f(5, 4)
>>> 3.0

f(x=13, y=5)
>>> 12.0
```

____________________________________


### Vectors
To declare a vector, just type a semicolon ; or a vertical bar |. This will separate 
the equation parts from the nearest left bracket to the right. Vectors can consist of 
only numbers or all equation types. Filling a vector only with numbers speeds up calculations 
considerably.
```python
a = parser.parse("(1| 2| 3)")         # fast Vector
b = parser.parse("[x; y; 3*z]")	      # square brackets can be used interchangeably
parser.set_param("c", "3*3; 4 | z")   # they don't have to be surrounded by brackets
v = parser.parse("a + r*b + s*c")
v.set_vars(x=1, y=2, z=1)
v(3, 2)
>>> [22. 16. 14.]
```

____________________________________

### Adding, Multiplying, Concatenating...
There are multiple methods which allow you to concatenate equations in various way.
You can directly add them, or parse them added together as a new equations. Consider that 
such an action merges the variable dictionaries of all involved functions.
```python
f = parser.parse("x^2")
g = parser.parse("y")
f.var, g.var
>>> {'x': Number(value=nan)} {'y': Number(value=nan)}

new_eq = parser.parse("f+g")
new_eq = f + g		# nearly equivalent to above
new_eq
>>> x ^ 2.0 + y

new_eq.var
>>> {'x': Number(value=nan), 'y': Number(value=nan)}

f.var, g.var	# vars have been merged
>>> {'x': Number(value=nan), 'y': Number(value=nan)} {'x': Number(value=nan), 'y': Number(value=nan)}
```
The difference between parsing the equation concatenation or directly concatenating them is that 
a direct concatenation does not register the new equation in the parser, such that it can not be used 
in further parsing.

____________________________________

### Operators
You can have a look at all operators by typing `print(eq.operators)`. It shows you a dictionary of all operator 
strings as well as their number of input arguments, precedences and corresponding functions. There are all basic 
operators (+, -, \*, /, :, ^), logic operators, elementary functions, trigonometric functions, hyperbolic functions, 
rounding functions, vector functions and some special ones.

The sum and product functions can either return the product or sum of a vector or can used like the corresponding mathematical 
symbols with a count variable, a stop value and a function. In order to use it as the last, you have to pass a three element 
vector to `sum` with an assignment as the first argument.
vector to `sum` or `prod` with an assignment as the first argument.
```python
# sum of a vector
vector_sum = parser.parse("sum(1; 2; 4; 8; 16; 32)")

# define the Riemann zeta function
zeta = parser.parse("sum(n=1;inf; 1/n^s)")
zeta(s=2)
>>> 1.6449330668497266
parser.parse("pi^2/6").value
>>> 1.6449340668482264

# product of a vector
vector_product = parser.parse("prod(1-1/2^2; 1-1/3^2; 1-1/5^2)")
vector_product.value
>>> 0.64

# define the factorial
fact = parser.parse("prod(k=1;n; k)")
parser.parse("fact(n = 10)").value	# or just fact(n=10)
>>> 3628800.0
```

