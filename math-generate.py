import random
import operator

OUTPUT_FILE = "math.txt"
NUM_SAMPLES = 500_000

OPS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
}


def rand_num():
    return random.randint(-1000, 1000)


def safe_pow():
    base = random.randint(-15, 15)
    exp = random.randint(0, 5)
    return f"{base} ** {exp}", base ** exp


def safe_div():
    a = rand_num()
    b = 0
    while b == 0:
        b = random.randint(-100, 100)
    return f"{a} // {b}", a // b


def safe_mod():
    a = rand_num()
    b = 0
    while b == 0:
        b = random.randint(-100, 100)
    return f"{a} % {b}", a % b


def binary():
    op = random.choice(["+", "-", "*"])

    a = rand_num()
    b = rand_num()

    expr = f"{a} {op} {b}"
    ans = OPS[op](a, b)

    return expr, ans


def three_term():
    op1 = random.choice(["+", "-", "*"])
    op2 = random.choice(["+", "-", "*"])

    a = rand_num()
    b = rand_num()
    c = rand_num()

    expr = f"{a} {op1} {b} {op2} {c}"
    ans = eval(expr)

    return expr, ans


def parentheses():
    op1 = random.choice(["+", "-", "*"])
    op2 = random.choice(["+", "-", "*"])

    a = rand_num()
    b = rand_num()
    c = rand_num()

    if random.random() < 0.5:
        expr = f"({a} {op1} {b}) {op2} {c}"
    else:
        expr = f"{a} {op1} ({b} {op2} {c})"

    ans = eval(expr)
    return expr, ans


def nested():
    op1 = random.choice(["+", "-", "*"])
    op2 = random.choice(["+", "-", "*"])
    op3 = random.choice(["+", "-", "*"])

    a = rand_num()
    b = rand_num()
    c = rand_num()
    d = rand_num()

    expr = f"(({a} {op1} {b}) {op2} {c}) {op3} {d}"
    ans = eval(expr)

    return expr, ans


def unary():
    a = rand_num()
    b = rand_num()

    expr = f"-({a}) + {b}"
    ans = eval(expr)

    return expr, ans


generators = [
    binary,
    three_term,
    parentheses,
    nested,
    unary,
    safe_div,
    safe_mod,
    safe_pow,
]

with open(OUTPUT_FILE, "w") as f:
    for i in range(NUM_SAMPLES):
        g = random.choice(generators)

        expr, ans = g()

        f.write(f"{expr} = {ans}\n")

        if (i + 1) % 10000 == 0:
            print(f"{i+1:,} samples generated")

print("Done!")