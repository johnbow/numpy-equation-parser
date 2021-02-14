# numpy-equation-parser
Modules to quickly parse and use functions for graphing and computation.

## Installation
Just copy the eq module into your project. Then you can import it with `from eq import eq`.
For using the on-the-fly name parsing feature, you have to install
the [varname](https://github.com/pwwang/python-varname) module by 
typing `pip install -U varname`.

## Examples
Here are an example code snippet that shows the module's functionality.

```java
parser = eq.EqParser()
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

## Usage

### Parser
Instantiate the parser directly. All parameters and equations belong to the parser and can 
be retrieved by their names with the syntax `parser[name]`. Equations are stored in a tree structure 
for fast calculation purposes.
```python
from eq import eq

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

