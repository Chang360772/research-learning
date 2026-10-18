import numpy as np

P = np.array([0.2,0.5,0.3])
Q = np.array([0.3,0.4,0.3])
Q_new = np.array([0.5,0.4,0.1])
Q_new_new = np.array([0.01,0.001,0.9989])

kl_pq = np.sum(P*np.log(P/Q)) #0.030478754035472025
kl_qp = np.sum(Q*np.log(Q/P)) #0.032382111906765376
kl_pq_new = np.sum(P*np.log(P/Q_new)) #0.2478973158827067
kl_qp_new = np.sum(Q_new*np.log(Q_new/P)) #0.25902671654458276
kl_pq_new_new = np.sum(P*np.log(P/Q_new_new)) #3.345588844257323
kl_qp_new_new = np.sum(Q_new_new*np.log(Q_new_new/P)) #1.1653771086291709


print(kl_pq,kl_qp)
print(kl_pq_new,kl_qp_new)
print(kl_pq_new_new,kl_qp_new_new)
#结论：KL 一定非负，只有P=Q时才是0，而且其具有不对称性，即kl.pq != kl.qp