with open("blc_pre.nql") as a, open("loader.blc") as b, open("blc.nql", "w") as c:
    a2 = a.read()
    b2 = b.read().split("\n")[-1]
    prog = []
    prefix = ""
    vardepth = 0
    for i in b2:
        if prefix == "":
            prefix = i
            vardepth = 0
        elif prefix == "1":
            if i == "1":
                vardepth += 1
            elif i == "0":
                prog.append((0,vardepth))
                prefix = ""
                while 1:
                    if len(prog) >= 2 and prog[-2] == (-1,) and prog[-1][0] >= 0:
                        x = prog.pop()
                        y = prog.pop()
                        prog.append((1,x))
                    elif len(prog) >= 3 and prog[-3] == (-2,) and prog[-1][0] >= 0 and prog[-2][0] >= 0:
                        x = prog.pop() # arg
                        y = prog.pop() # fun
                        z = prog.pop()
                        prog.append((2,y,x))
                    else:
                        break
        elif prefix == "0":
            if i == "0":
                prog.append((-1,))
            elif i == "1":
                prog.append((-2,))
            prefix = ""
    assert len(prog) == 1
    def emit(x):
        if x[0] == 0:
            # every push_x zeroes t2, and it starts at 0
            if x[1] == 0:
                return "push_var(t2);"
            return f"t2 = t2 + {x[1]}; push_var(t2);"
        elif x[0] == 1:
            return f"{emit(x[1])} push_lam();"
        elif x[0] == 2:
            return f"{emit(x[1])} {emit(x[2])} push_app();"
    c.write(a2.replace("/* program goes here */", emit(prog[0])))