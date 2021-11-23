from mpi4py import MPI
import numpy as np
import time
start_time=time.time()
COMM = MPI.COMM_WORLD
my_pe_num = COMM.Get_rank()
def output(data):
	data.tofile("plate.out")
max_iterations = 5000
dt1 = 100
df2 =100
dt3 =100
dt4 =100
dt =100
iteration = 1
#init
MAX_TEMP_ERROR=3
for i in range (4):
	COMM.barrier()
	if i==my_pe_num:
		master=np.empty((252, 1002))
		master_last=np.empty((252,1002))
		master_last[:,:]=0
		for j in range (252):
			#set right side boundary
			master_last[j,1001] = 0.1*(j+250*i)
#set boundary for pe 3 which includes bottom
if my_pe_num==3:
	for k in range (1001):
		master_last[251, k] = 0.1*k
	#according to the context [1001,1001] is 0, it's a empty coordinates.
	master_last[251,1001] = 0
dt=100
while ( dt > MAX_TEMP_ERROR ) and ( iteration < max_iterations ):
	COMM.barrier()
	dt=0
	for i in range (4):
		if i==my_pe_num:
			for j in range( 1 , 251 ):
				for k in range( 1 , 1001 ):
					master[j,k] = 0.25 * ( master_last[j+1,k] + master_last[j-1,k] + master_last[j,k+1] + master_last[j,k-1]  )
			for j in range (1, 251):
				for k in range(1, 1001 ):
					dt=max(dt, master[j,k] - master_last[j,k])
					master_last[j,k] = master[j,k]

	#from thermodynamics PE3 should have largest dt and should update to other PE
	dt=COMM.reduce(dt,op=MPI.MAX, root=0)
	dt=COMM.bcast(dt, root=0)
	#communicate
	#send & recv
	for i in range (3):
		if i==my_pe_num:
			MPI.COMM_WORLD.Send(master_last[250,:], dest=i+1, tag=1)
			MPI.COMM_WORLD.Recv(master_last[251,:], source = i+1, tag= 1)
		if i+1==my_pe_num:
			MPI.COMM_WORLD.Send(master_last[1,:], dest=i, tag=1)
			MPI.COMM_WORLD.Recv(master_last[0,:], source = i, tag= 1)
	#one pe print is enough
	if my_pe_num==0:
		print(iteration, dt)
	iteration=iteration+1
master_final=COMM.gather(master_last,root=0)
#one pe is enough to do this
if my_pe_num==0:
	master_final=np.concatenate(master_final)
	#delete ghost cells
	master_final=np.delete(master_final,[251,252,503,504,755,756], axis=0)
	output(master_final)
	end_time=time.time()
	print("Total execution time: {} seconds".format(end_time - start_time))

