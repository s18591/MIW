import numpy as np
import matplotlib.pyplot as plt
if __name__ == '__main__':
    l = 0
    w = 0
    t = 0
    start = ['Sc', 'P', 'S']
    start_dict = {'P': 0, 'S': 1, 'Sc': 2}
    state = ''
    p_s_start = np.array([1, 1, 1])
    p_p_start = np.array([1, 1, 1])
    p_sc_start = np.array([1, 1, 1])
    l_inputs = ['']
    Wins = [0]
    Loses = [0]
    Ties = [0]

    while True:
        input1 = input()
        l_inputs.append(input1)
        lastElement = l_inputs[len(l_inputs) - 2]

        if input1 == 'q':
            plt.plot(range(len(Wins)), Wins, label='Wins', alpha=0.5)
            plt.plot(range(len(Wins)), Loses, label='Loses', alpha=0.5)
            plt.plot(range(len(Wins)), Ties, label='Ties', alpha=0.5)
            plt.legend()
            plt.show()
            break

        if lastElement == '':
            state = np.random.choice(start, p=p_s_start / sum(p_s_start))
        if lastElement == 'P':
            p_p_start[start_dict[input1]] = p_p_start[start_dict[input1]] + 1
            state = np.random.choice(start, p=p_p_start / sum(p_p_start))
        if lastElement == 'S':
            p_s_start[start_dict[input1]] = p_s_start[start_dict[input1]] + 1
            state = np.random.choice(start, p=p_s_start / sum(p_s_start))
        if lastElement == 'Sc':
            p_sc_start[start_dict[input1]] = p_sc_start[start_dict[input1]] + 1
            state = np.random.choice(start, p=p_sc_start / sum(p_sc_start))

        if input1 == 'P' and state == 'S':
            w = w+1
            Wins.append(w)
            Ties.append(t)
            Loses.append(l)
            print(state)
            print(f"W =  {w}  L =  {l}   T =  {t}")
        elif input1 == 'S' and state == 'Sc':
            w = w+1
            Wins.append(w)
            Ties.append(t)
            Loses.append(l)
            print(state)
            print(f"W =  {w}  L =  {l}   T =  {t}")
        elif input1 == 'Sc' and state == 'P':
            w = w+1
            Wins.append(w)
            Ties.append(t)
            Loses.append(l)
            print(state)
            print(f"W =  {w}  L =  {l}   T =  {t}")
        elif input1 == state:
            t = t+1
            Wins.append(w)
            Ties.append(t)
            Loses.append(l)
            print(state)
            print(f"W =  {w}  L =  {l}   T =  {t}")
        else:
            l = l+1
            Wins.append(w)
            Ties.append(t)
            Loses.append(l)
            print(state)
            print(f"W =  {w}  L =  {l}   T =  {t}")
