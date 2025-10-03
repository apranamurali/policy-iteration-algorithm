# POLICY ITERATION ALGORITHM

## AIM
The goal of the notebook is to implement and evaluate a policy iteration algorithm within a custom environment (gym-walk) to find the optimal policy that maximizes the agent's performance in terms of reaching a goal state with the highest probability and reward.
## PROBLEM STATEMENT
The task is to develop and apply a policy iteration algorithm to solve a grid-based environment (gym-walk). The environment consists of states the agent must navigate through to reach a goal. The agent has to learn the best sequence of actions (policy) that maximizes its chances of reaching the goal state while obtaining the highest cumulative reward.
## POLICY ITERATION ALGORITHM
Initialize: Start with a random policy for each state and initialize the value function arbitrarily.

Policy Evaluation: For each state, evaluate the current policy by computing the expected value function under the current policy.

Policy Improvement: Improve the policy by making it greedy with respect to the current value function (i.e., choose the action that maximizes the value function for each state).

Check Convergence: Repeat the evaluation and improvement steps until the policy stabilizes (i.e., when no further changes to the policy occur).

Optimal Policy: Once convergence is achieved, the policy is considered optimal, providing the best actions for the agent in each state.

## POLICY IMPROVEMENT FUNCTION
### Name : APARNA.M
### Register Number : 212223220008
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
      for a in range(len(P[s])):
        for prob, next_state, reward, done in P[s][a]:
          Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return new_pi

pi_2 = policy_improvement(V1, P)
print("Name: APARNA.M")
print("Register Number: 212223220008")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

## POLICY ITERATION FUNCTION
### NAME : APARNA.M
### REGISTER NUMBER : 212223220008
def policy_iteration(P, gamma=1.0, theta=1e-10):
  random_actions = np.random.choice(tuple(P[0].keys()), len(P))
  pi = lambda s: {s:a for s, a in enumerate(random_actions)}[s]

  while True:
    old_pi = {s: pi(s) for s in range(len(P))}
    V = policy_evaluation(pi, P, gamma, theta)
    pi = policy_improvement(V, P, gamma)

    if old_pi == {s: pi(s) for s in range(len(P))}:
      break

  return V, pi
optimal_V, optimal_pi = policy_iteration(P)
print("Name: APARNA.M")
print("Register Number: 212223220008")
print('Optimal policy and state-value function (PI):')
print_policy(optimal_pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)



## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
## POLICY


<img width="840" height="169" alt="image" src="https://github.com/user-attachments/assets/7260bfd6-ba1f-4b67-a390-0cf9c1a8f06f" />


## STATE VALUE FUNCTION


<img width="774" height="192" alt="image" src="https://github.com/user-attachments/assets/443335df-2e70-42f2-b2c8-878c1ce55362" />


## SUCCESS


<img width="1167" height="61" alt="image" src="https://github.com/user-attachments/assets/1eb4e973-58d6-4756-b05a-6adbd0c7abb1" />



### 2. Policy, Value function and success rate for the Improved Policy

## POLICY


<img width="781" height="200" alt="image" src="https://github.com/user-attachments/assets/cce68d2c-13f2-40a1-bb0a-778ffc3b5607" />


## STATE VALUE FUNCTION

<img width="908" height="189" alt="image" src="https://github.com/user-attachments/assets/f7e61fdc-4205-401c-a825-7be7a1c436a5" />


## SUCCESS

<img width="1002" height="60" alt="image" src="https://github.com/user-attachments/assets/3c189924-07a7-4255-ab51-5743bc1e6b27" />


<img width="872" height="60" alt="image" src="https://github.com/user-attachments/assets/6c9d71f2-0b55-4b69-acf6-0c9b9f97e3bd" />





### 3. Policy, Value function and success rate after policy iteration

## POLICY


<img width="909" height="234" alt="image" src="https://github.com/user-attachments/assets/c88efe5f-2244-41db-bb42-63d94bdbb843" />


## STATE VALUE FUNCTION


<img width="796" height="203" alt="image" src="https://github.com/user-attachments/assets/ec8d3e61-6c03-45f4-ab73-2f6115dbc11a" />


## SUCCESS


<img width="1029" height="60" alt="image" src="https://github.com/user-attachments/assets/debdb9c5-c391-4470-b0d7-79f029f8a975" />



## RESULT:
Thus the program to iterate the policy evaluation and policy improvement is executed successfully.
