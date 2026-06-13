#!/usr/bin/env python3
import re

src = open("/Users/aleksandersekkelsten/thesis/oral_exam_notes.md", encoding="utf-8").read()

# ---- 1. protect math and code with placeholders -------------------------
store = {}
ctr = [0]


def tok(kind):
    ctr[0] += 1
    return f"@@{kind}{ctr[0]}@@"


def esc_code(s):
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("#", r"\#")
        .replace("$", r"\$")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("_", r"\_")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


# display math $$ ... $$
def repl_dmath(m):
    t = tok("DMATH")
    store[t] = "\\[" + m.group(1).strip() + "\\]"
    return t


src = re.sub(r"\$\$(.+?)\$\$", repl_dmath, src, flags=re.DOTALL)


# inline code `...`
def repl_code(m):
    t = tok("CODE")
    store[t] = r"\texttt{" + esc_code(m.group(1)) + "}"
    return t


src = re.sub(r"`([^`]+)`", repl_code, src)


# inline math $ ... $
def repl_imath(m):
    t = tok("IMATH")
    store[t] = "$" + m.group(1) + "$"
    return t


src = re.sub(r"\$([^$]+?)\$", repl_imath, src)

# ---- 2. inline text processing ------------------------------------------
UNI = {
    "—": "---",
    "–": "--",
    "§": r"\S{}",
    "→": r"$\to$",
    "↔": r"$\leftrightarrow$",
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "…": r"\ldots{}",
    "·": r"$\cdot$",
    "²": r"$^{2}$",
    "ö": r"\"o",
    "é": r"\'e",
    "ę": r"\k{e}",
    "✓": r"$\checkmark$",
    "✗": r"$\times$",
    "×": r"$\times$",
    "≈": r"$\approx$",
    "∘": r"$\circ$",
}


def esc_text(s):
    # escape LaTeX specials FIRST (unicode chars are untouched by this)...
    s = (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )
    # ...then map unicode to LaTeX so the inserted backslashes survive
    for u, r in UNI.items():
        s = s.replace(u, r)
    return s


def inline(s):
    s = esc_text(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"\*(.+?)\*", r"\\emph{\1}", s)
    return s


# ---- 3. block parsing ----------------------------------------------------
out = []
stack: list[tuple[int, str]] = []


def close_all():
    while stack:
        out.append("\\end{" + stack[-1][1] + "}")
        stack.pop()


def close_to(indent):
    while stack and stack[-1][0] > indent:
        out.append("\\end{" + stack[-1][1] + "}")
        stack.pop()


lines = src.split("\n")
HEAD = re.compile(r"^(#{1,6})\s+(.*)$")
UL = re.compile(r"^(\s*)[-*]\s+(.*)$")
OL = re.compile(r"^(\s*)\d+\.\s+(.*)$")

for ln in lines:
    m = HEAD.match(ln)
    if m:
        close_all()
        lvl = len(m.group(1))
        txt = inline(m.group(2).strip())
        cmd = {1: r"\section*", 2: r"\subsection*", 3: r"\subsubsection*"}.get(lvl)
        if cmd:
            out.append(cmd + "{" + txt + "}")
        else:
            out.append(r"\paragraph{" + txt + "}\\mbox{}\\\\")
        continue
    if re.match(r"^---+\s*$", ln):
        close_all()
        out.append(r"\vspace{4pt}\hrule\vspace{8pt}")
        continue
    mul = UL.match(ln)
    mol = OL.match(ln)
    active = mul or mol
    if active:
        env = "itemize" if mul else "enumerate"
        indent = len(active.group(1))
        content = inline(active.group(2).strip())
        close_to(indent)
        if stack and stack[-1][0] == indent and stack[-1][1] != env:
            out.append("\\end{" + stack[-1][1] + "}")
            stack.pop()
        if not stack or stack[-1][0] < indent:
            out.append("\\begin{" + env + "}")
            stack.append((indent, env))
        out.append("\\item " + content)
        continue
    if ln.strip() == "":
        if not stack:
            out.append("")  # paragraph break
        continue
    # plain text line
    leading = len(ln) - len(ln.lstrip(" "))
    if stack and leading == 0:
        close_all()
    out.append(inline(ln.strip()))

close_all()
body = "\n".join(out)

# ---- 4. restore placeholders --------------------------------------------
for t, v in store.items():
    body = body.replace(t, v)

# ---- 5. wrap in a document ----------------------------------------------
PRE = r"""\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[a4paper,margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{enumitem}
\usepackage{microtype}
\usepackage[dvipsnames]{xcolor}
\usepackage{parskip}
\usepackage{titlesec}
\setlist{leftmargin=1.4em,itemsep=1pt,topsep=2pt}
\titleformat{\section}{\Large\bfseries\color{NavyBlue}}{}{0pt}{}
\titleformat{\subsection}{\large\bfseries\color{BrickRed}}{}{0pt}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{}{0pt}{}
\setcounter{secnumdepth}{0}
\sloppy
\begin{document}
"""
POST = "\n\\end{document}\n"
open("/tmp/oral_exam_notes.tex", "w", encoding="utf-8").write(PRE + body + POST)
print("wrote /tmp/oral_exam_notes.tex", len(body), "chars body")
