"""
ToyLang Mini-Compiler: Lexer, Parser, 3-Address Code (IR), Virtual Machine, and Optimizations.
Track F — Compiler Optimization.
"""
import re, time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# =================================================================
# 1. Lexer
# =================================================================
TOKEN_SPEC = [
    ('NUMBER',   r'\d+'),
    ('WHILE',    r'while\b'),
    ('PRINT',    r'print\b'),
    ('ID',       r'[a-zA-Z_][a-zA-Z0-9_]*'),
    ('OP',       r'[+\-*/]'),
    ('ASSIGN',   r'='),
    ('LPAREN',   r'\('),
    ('RPAREN',   r'\)'),
    ('LBRACE',   r'\{'),
    ('RBRACE',   r'\}'),
    ('SEMI',     r';'),
    ('SKIP',     r'[ \t\n]+'),
    ('MISMATCH', r'.'),
]
TOKEN_RE = '|'.join('(?P<%s>%s)' % pair for pair in TOKEN_SPEC)

def tokenize(code):
    tokens = []
    for mo in re.finditer(TOKEN_RE, code):
        kind = mo.lastgroup
        value = mo.group()
        if kind == 'NUMBER':
            tokens.append(('NUMBER', int(value)))
        elif kind == 'SKIP':
            continue
        elif kind == 'MISMATCH':
            raise SyntaxError(f"Unexpected character {value!r}")
        else:
            tokens.append((kind, value))
    return tokens

# =================================================================
# 2. Parser & AST
# =================================================================
class ASTNode: pass

class NumberNode(ASTNode):
    def __init__(self, val): self.val = val

class VariableNode(ASTNode):
    def __init__(self, name): self.name = name

class BinOpNode(ASTNode):
    def __init__(self, op, left, right):
        self.op, self.left, self.right = op, left, right

class AssignNode(ASTNode):
    def __init__(self, var, expr):
        self.var, self.expr = var, expr

class WhileNode(ASTNode):
    def __init__(self, cond, body):
        self.cond, self.body = cond, body

class PrintNode(ASTNode):
    def __init__(self, var):
        self.var = var

class Parser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else (None, None)

    def consume(self, expected_kind=None):
        kind, val = self.peek()
        if expected_kind and kind != expected_kind:
            raise SyntaxError(f"Expected {expected_kind}, got {kind}")
        self.pos += 1
        return val

    def parse(self):
        stmts = []
        while self.peek()[0] is not None:
            stmts.append(self.parse_statement())
        return stmts

    def parse_statement(self):
        kind, val = self.peek()
        if kind == 'ID':
            var = self.consume('ID')
            self.consume('ASSIGN')
            expr = self.parse_expr()
            self.consume('SEMI')
            return AssignNode(var, expr)
        elif kind == 'PRINT':
            self.consume('PRINT')
            self.consume('LPAREN')
            var = self.consume('ID')
            self.consume('RPAREN')
            self.consume('SEMI')
            return PrintNode(var)
        elif kind == 'WHILE':
            self.consume('WHILE')
            self.consume('LPAREN')
            cond = self.consume('ID')
            self.consume('RPAREN')
            self.consume('LBRACE')
            body = []
            while self.peek()[0] != 'RBRACE':
                body.append(self.parse_statement())
            self.consume('RBRACE')
            return WhileNode(cond, body)
        else:
            raise SyntaxError(f"Unexpected statement starting with {kind}")

    def parse_expr(self):
        node = self.parse_term()
        while self.peek()[0] == 'OP' and self.peek()[1] in ('+', '-'):
            op = self.consume('OP')
            node = BinOpNode(op, node, self.parse_term())
        return node

    def parse_term(self):
        node = self.parse_factor()
        while self.peek()[0] == 'OP' and self.peek()[1] in ('*', '/'):
            op = self.consume('OP')
            node = BinOpNode(op, node, self.parse_factor())
        return node

    def parse_factor(self):
        kind, val = self.peek()
        if kind == 'NUMBER':
            self.consume('NUMBER')
            return NumberNode(val)
        elif kind == 'ID':
            self.consume('ID')
            return VariableNode(val)
        elif kind == 'LPAREN':
            self.consume('LPAREN')
            expr = self.parse_expr()
            self.consume('RPAREN')
            return expr
        raise SyntaxError(f"Unexpected factor {kind}")

# =================================================================
# 3. Intermediate Representation (3-Address Code)
# =================================================================
class Instr:
    def __init__(self, op, arg1=None, arg2=None, res=None):
        self.op = op          # ADD, SUB, MUL, DIV, ASSIGN, PRINT, LABEL, JUMP, JUMP_IF_ZERO
        self.arg1 = arg1
        self.arg2 = arg2
        self.res = res

    def __repr__(self):
        if self.op in ('ADD', 'SUB', 'MUL', 'DIV'):
            return f"{self.res} = {self.arg1} {self.op} {self.arg2}"
        elif self.op == 'ASSIGN':
            return f"{self.res} = {self.arg1}"
        elif self.op == 'PRINT':
            return f"PRINT {self.arg1}"
        elif self.op == 'LABEL':
            return f"LABEL {self.arg1}:"
        elif self.op == 'JUMP':
            return f"JUMP {self.arg1}"
        elif self.op == 'JUMP_IF_ZERO':
            return f"IF {self.arg1} == 0 GOTO {self.arg2}"
        return f"{self.op} {self.arg1} {self.arg2} -> {self.res}"

class IRGenerator:
    def __init__(self):
        self.instructions = []
        self.temp_count = 0
        self.label_count = 0

    def new_temp(self):
        self.temp_count += 1
        return f"t{self.temp_count}"

    def new_label(self):
        self.label_count += 1
        return f"L{self.label_count}"

    def generate(self, ast_stmts):
        for stmt in ast_stmts:
            self.gen_stmt(stmt)
        return self.instructions

    def gen_stmt(self, stmt):
        if isinstance(stmt, AssignNode):
            res = self.gen_expr(stmt.expr)
            self.instructions.append(Instr('ASSIGN', res, None, stmt.var))
        elif isinstance(stmt, PrintNode):
            self.instructions.append(Instr('PRINT', stmt.var))
        elif isinstance(stmt, WhileNode):
            lbl_start = self.new_label()
            lbl_end = self.new_label()
            self.instructions.append(Instr('LABEL', lbl_start))
            self.instructions.append(Instr('JUMP_IF_ZERO', stmt.cond, lbl_end))
            for s in stmt.body:
                self.gen_stmt(s)
            self.instructions.append(Instr('JUMP', lbl_start))
            self.instructions.append(Instr('LABEL', lbl_end))

    def gen_expr(self, expr):
        if isinstance(expr, NumberNode):
            t = self.new_temp()
            self.instructions.append(Instr('ASSIGN', expr.val, None, t))
            return t
        elif isinstance(expr, VariableNode):
            return expr.name
        elif isinstance(expr, BinOpNode):
            t1 = self.gen_expr(expr.left)
            t2 = self.gen_expr(expr.right)
            res = self.new_temp()
            op_map = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV'}
            self.instructions.append(Instr(op_map[expr.op], t1, t2, res))
            return res

# =================================================================
# 4. Virtual Machine Interpreter
# =================================================================
class VirtualMachine:
    def __init__(self):
        self.env = {}
        self.output = []
        self.instruction_exec_count = 0

    def run(self, instructions, max_cycles=100000):
        pc = 0
        labels = {}
        for i, instr in enumerate(instructions):
            if instr.op == 'LABEL':
                labels[instr.arg1] = i

        self.instruction_exec_count = 0
        while pc < len(instructions):
            self.instruction_exec_count += 1
            if self.instruction_exec_count > max_cycles:
                raise RuntimeError("Execution cycle limit exceeded (infinite loop)")

            instr = instructions[pc]
            if instr.op == 'LABEL':
                pc += 1
            elif instr.op == 'ASSIGN':
                val = instr.arg1 if isinstance(instr.arg1, int) else self.env.get(instr.arg1, 0)
                self.env[instr.res] = val
                pc += 1
            elif instr.op in ('ADD', 'SUB', 'MUL', 'DIV'):
                v1 = instr.arg1 if isinstance(instr.arg1, int) else self.env.get(instr.arg1, 0)
                v2 = instr.arg2 if isinstance(instr.arg2, int) else self.env.get(instr.arg2, 0)
                if instr.op == 'ADD': self.env[instr.res] = v1 + v2
                elif instr.op == 'SUB': self.env[instr.res] = v1 - v2
                elif instr.op == 'MUL': self.env[instr.res] = v1 * v2
                elif instr.op == 'DIV': self.env[instr.res] = v1 // v2 if v2 != 0 else 0
                pc += 1
            elif instr.op == 'PRINT':
                val = instr.arg1 if isinstance(instr.arg1, int) else self.env.get(instr.arg1, 0)
                self.output.append(val)
                pc += 1
            elif instr.op == 'JUMP':
                pc = labels[instr.arg1] + 1
            elif instr.op == 'JUMP_IF_ZERO':
                val = instr.arg1 if isinstance(instr.arg1, int) else self.env.get(instr.arg1, 0)
                if val == 0:
                    pc = labels[instr.arg2] + 1
                else:
                    pc += 1
        return self.output, self.env

# =================================================================
# 5. Optimization Passes
# =================================================================
# Pass 1: Constant Folding & Propagation
def pass_constant_folding(instructions):
    consts = {}
    new_ir = []
    for instr in instructions:
        # Resolve operands
        a1 = consts.get(instr.arg1, instr.arg1)
        a2 = consts.get(instr.arg2, instr.arg2)
        if instr.op == 'ASSIGN':
            if isinstance(a1, int):
                consts[instr.res] = a1
            elif instr.res in consts:
                del consts[instr.res]
            new_ir.append(Instr('ASSIGN', a1, None, instr.res))
        elif instr.op in ('ADD', 'SUB', 'MUL', 'DIV'):
            if isinstance(a1, int) and isinstance(a2, int):
                if instr.op == 'ADD': val = a1 + a2
                elif instr.op == 'SUB': val = a1 - a2
                elif instr.op == 'MUL': val = a1 * a2
                elif instr.op == 'DIV': val = a1 // a2 if a2 != 0 else 0
                consts[instr.res] = val
                new_ir.append(Instr('ASSIGN', val, None, instr.res))
            else:
                if instr.res in consts:
                    del consts[instr.res]
                new_ir.append(Instr(instr.op, a1, a2, instr.res))
        else:
            if instr.op == 'LABEL':
                consts.clear()  # reset across control flow boundaries
            new_ir.append(Instr(instr.op, a1, a2, instr.res))
    return new_ir

# Pass 2: Strength Reduction
def pass_strength_reduction(instructions):
    new_ir = []
    for instr in instructions:
        if instr.op == 'MUL':
            # x * 2 -> x + x
            if instr.arg2 == 2:
                new_ir.append(Instr('ADD', instr.arg1, instr.arg1, instr.res))
            elif instr.arg1 == 2:
                new_ir.append(Instr('ADD', instr.arg2, instr.arg2, instr.res))
            # x * 0 -> 0
            elif instr.arg2 == 0 or instr.arg1 == 0:
                new_ir.append(Instr('ASSIGN', 0, None, instr.res))
            # x * 1 -> x
            elif instr.arg2 == 1:
                new_ir.append(Instr('ASSIGN', instr.arg1, None, instr.res))
            elif instr.arg1 == 1:
                new_ir.append(Instr('ASSIGN', instr.arg2, None, instr.res))
            else:
                new_ir.append(instr)
        else:
            new_ir.append(instr)
    return new_ir

# Pass 3: Common Subexpression Elimination (CSE)
def pass_cse(instructions):
    expr_map = {}
    new_ir = []
    for instr in instructions:
        if instr.op in ('ADD', 'SUB', 'MUL', 'DIV'):
            key = (instr.op, instr.arg1, instr.arg2)
            if key in expr_map:
                prev_temp = expr_map[key]
                new_ir.append(Instr('ASSIGN', prev_temp, None, instr.res))
            else:
                expr_map[key] = instr.res
                new_ir.append(instr)
        else:
            if instr.op == 'LABEL':
                expr_map.clear()
            new_ir.append(instr)
    return new_ir

# Pass 4: Dead Code Elimination (DCE)
def pass_dce(instructions):
    # Find all used variables
    used = set()
    for instr in instructions:
        if instr.op != 'LABEL':
            if isinstance(instr.arg1, str): used.add(instr.arg1)
            if isinstance(instr.arg2, str): used.add(instr.arg2)

    new_ir = []
    for instr in instructions:
        # Keep instruction if it modifies control flow, prints, or result is used
        if instr.op in ('PRINT', 'LABEL', 'JUMP', 'JUMP_IF_ZERO'):
            new_ir.append(instr)
        elif instr.res in used:
            new_ir.append(instr)
    return new_ir

# Pass 5: Loop-Invariant Code Motion (LICM)
def pass_licm(instructions):
    # Identify loops between LABEL L_start and JUMP L_start
    loop_start_idx = None
    loop_end_idx = None
    for i, instr in enumerate(instructions):
        if instr.op == 'LABEL' and loop_start_idx is None:
            loop_start_idx = i
        elif instr.op == 'JUMP' and loop_start_idx is not None:
            loop_end_idx = i
            break

    if loop_start_idx is None or loop_end_idx is None:
        return instructions

    # Find variables modified inside the loop
    modified_in_loop = set()
    for i in range(loop_start_idx, loop_end_idx + 1):
        if instructions[i].res is not None:
            modified_in_loop.add(instructions[i].res)

    invariants = []
    body_instrs = []
    for i in range(loop_start_idx, loop_end_idx + 1):
        instr = instructions[i]
        # Invariant if computation only uses variables not modified inside loop
        if instr.op in ('ADD', 'SUB', 'MUL', 'DIV', 'ASSIGN'):
            uses = set()
            if isinstance(instr.arg1, str): uses.add(instr.arg1)
            if isinstance(instr.arg2, str): uses.add(instr.arg2)
            if not (uses & modified_in_loop):
                invariants.append(instr)
                continue
        body_instrs.append(instr)

    # Place invariants right before loop header
    new_ir = instructions[:loop_start_idx] + invariants + body_instrs + instructions[loop_end_idx + 1:]
    return new_ir

def optimize_all(ir):
    t0 = time.perf_counter()
    code = ir
    code = pass_constant_folding(code)
    code = pass_strength_reduction(code)
    code = pass_cse(code)
    code = pass_licm(code)
    code = pass_dce(code)
    compile_time = time.perf_counter() - t0
    return code, compile_time

# =================================================================
# 6. Register Allocation: Greedy vs ILP
# =================================================================
def analyze_live_ranges(instructions):
    """Compute live ranges [start, end] for each variable in linear IR."""
    first_seen = {}
    last_seen = {}
    for idx, instr in enumerate(instructions):
        vars_in_instr = []
        if isinstance(instr.arg1, str) and not instr.arg1.startswith('L'): vars_in_instr.append(instr.arg1)
        if isinstance(instr.arg2, str) and not instr.arg2.startswith('L'): vars_in_instr.append(instr.arg2)
        if isinstance(instr.res, str): vars_in_instr.append(instr.res)
        for v in vars_in_instr:
            if v not in first_seen: first_seen[v] = idx
            last_seen[v] = idx
    intervals = {v: (first_seen[v], last_seen[v]) for v in first_seen}
    return intervals

def build_interference_graph(intervals):
    variables = sorted(list(intervals.keys()))
    n = len(variables)
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        s1, e1 = intervals[variables[i]]
        for j in range(i + 1, n):
            s2, e2 = intervals[variables[j]]
            # Overlap in live range implies interference
            if not (e1 < s2 or e2 < s1):
                adj[i, j] = 1
                adj[j, i] = 1
    return variables, adj

# Greedy Register Allocation (Chaitin-Briggs / Welsh-Powell heuristic)
def allocate_registers_greedy(variables, adj, k_registers):
    if len(variables) == 0:
        return {}, 0
    # Sort by descending degree
    degrees = np.sum(adj, axis=1)
    order = np.argsort(-degrees)
    color_map = {}
    spills = 0
    for node in order:
        v = variables[node]
        neighbor_colors = {color_map[variables[nbr]] for nbr in range(len(variables))
                           if adj[node, nbr] == 1 and variables[nbr] in color_map}
        available = [c for c in range(k_registers) if c not in neighbor_colors]
        if available:
            color_map[v] = min(available)
        else:
            color_map[v] = -1  # Spilled to memory
            spills += 1
    return color_map, spills

# ILP Register Allocation via scipy.optimize.milp
def allocate_registers_ilp(variables, adj, k_registers):
    n = len(variables)
    if n == 0:
        return {}, 0
    k = k_registers
    num_vars = n * k + n  # n*k register vars + n spill vars

    # Cost vector: 0 for register assignment, 1 for spill
    c = np.zeros(num_vars)
    for i in range(n):
        c[n * k + i] = 1.0  # penalty for spill

    # Constraints
    # 1. Assignment constraints: sum_c x[i,c] + s[i] == 1
    A_assign = np.zeros((n, num_vars))
    for i in range(n):
        for reg in range(k):
            A_assign[i, i * k + reg] = 1.0
        A_assign[i, n * k + i] = 1.0
    b_assign_eq = np.ones(n)

    # 2. Interference constraints: x[i,c] + x[j,c] <= 1
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 1]
    num_edge_cons = len(edges) * k
    A_inter = np.zeros((num_edge_cons, num_vars))
    row = 0
    for (i, j) in edges:
        for reg in range(k):
            A_inter[row, i * k + reg] = 1.0
            A_inter[row, j * k + reg] = 1.0
            row += 1
    b_inter_ub = np.ones(num_edge_cons)

    if num_edge_cons > 0:
        A_full = np.vstack([A_assign, A_inter])
        lhs = np.concatenate([b_assign_eq, np.full(num_edge_cons, -np.inf)])
        rhs = np.concatenate([b_assign_eq, b_inter_ub])
    else:
        A_full = A_assign
        lhs = b_assign_eq
        rhs = b_assign_eq

    constraints = LinearConstraint(A_full, lhs, rhs)
    integrality = np.ones(num_vars)
    bounds = Bounds(0, 1)

    res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
    if res.success:
        sol = np.round(res.x).astype(int)
        spills = int(round(res.fun))
        color_map = {}
        for i in range(n):
            assigned = False
            for reg in range(k):
                if sol[i * k + reg] == 1:
                    color_map[variables[i]] = reg
                    assigned = True
                    break
            if not assigned:
                color_map[variables[i]] = -1
        return color_map, spills
    return None, None
