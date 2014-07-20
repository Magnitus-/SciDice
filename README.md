SciDice
=======

Dice roller that accepts and expends upon an intuitive well established dice notation and can use various underlying distributions to generate results.

Requirements
============

- Python 2.7 or 3.x.

- numpy

- scipy

What Comes With It
==================

- A importable package for your python code

- A command-line script that allows you to type in dice formats and see the result when those dice are rolled

How To Install It
=================

Once you have the prerequisite (Python, numpy, scipy), just go in the directory containing the setup.py script and type: python setup.py install

I recommand that ubuntu users run it with sudo.

How To Use The Package In Your code
===================================

Make sure the package is installed if you want the example below to run it anywhere without tweaking.

```python
import SciDice
Test = SciDice.Dice(<FormatString>)
Rolls = Test.GenerateRolls()
```

For the format string, you can look at the unit test scripts for examples or if you are running ipython, you can type the following in your interpreter:

```python
import SciDice
SciDice.Dice?
```

How To Use The Command Line Script
==================================

Make sure the package is installed if you want to run the script anywhere without tweaking.

Type the following on the command line: SciDiceScript

You should see the following on the command line: SciDice>

After that, you can type in format strings like when you use the package in your code and see the results.

Additionally, after you typed in a format string, you can just hit enter subsequently to re-roll as many times as you wish.

To quit, you can type 'q', 'quit' or 'exit'.

I plan to add facilities with arguments to run the unit tests from the script in the future.
