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
