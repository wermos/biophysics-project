## Initial Setup

1. install Julia Lang
2. install Atom IDE
3. install uber-juno extension for Atom IDE
4. for windows add 'C:\Users\<username>\AppData\Local\Programs\Julia 1.5.3\bin' to environment variables > path
5. open project folder in Atom
6. Juno > environment > current folder
7. in Atom menu: Juno -> Open REPL

## Running Project

- Juno > environment > current folder OR execute this in REPL : Pkg.activate(pwd())
- Juno > working directory > current folder
- open .jl file to run
- to run entire code in file, in Atom menu: Juno -> Run All
- to run part of code, select lines and press Shift + Enter
- check Juno documentation for more features

# Using Package Manager

- to add package: ] add <package>
