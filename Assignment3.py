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
MAX_TEMP_ERROR=0.01
#master dict
master=dict()
master_last=dict()
for i in range (4):
	COMM.barrier()
	if i==my_pe_num:
		master[i]=np.empty((252, 1002))
		master_last[i]=np.empty((252,1002))
		master_last[i][:,:]=0
		for j in range (252):
			#set right side boundary
			master_last[i][j,1001] = 0.1*(j+250*i)
#set boundary for pe 3 which includes bottom
if my_pe_num==3:
	for k in range (1001):
		master_last[3][251, k] = 0.1*k
	#according to the context [1001,1001] is 0, it's a empty coordinates.
	master_last[3][251,1001] = 0
dt_final=100
dt=dict()
while ( dt_final > MAX_TEMP_ERROR ) and ( iteration < max_iterations ):
	dt[0]=0
	dt[1]=0
	dt[2]=0
	dt[3]=0
	for j in range( 1 , 251 ):
		for k in range( 1 , 1001 ):
			master[my_pe_num][j,k] = 0.25 * ( master_last[my_pe_num][j+1,k] + master_last[my_pe_num][j-1,k] + master_last[my_pe_num][j,k+1] + master_last[my_pe_num][j,k-1]  )
	for j in range (1, 251):
		for k in range(1, 1001 ):
			dt[my_pe_num]=max(dt[my_pe_num], master[my_pe_num][j,k] - master_last[my_pe_num][j,k])
			master_last[my_pe_num][j,k] = master[my_pe_num][j,k]
	COMM.barrier()
	#from thermodynamics PE3 should have largest dt and should update to other PE
	if my_pe_num==3:
		dt_final=max(dt[0],dt[1],dt[2],dt[3])
		#update other PE
		for i in range (3):
			MPI.COMM_WORLD.send(dt_final, dest=i, tag=1)
	else:
		dt_final=MPI.COMM_WORLD.recv( source = 3, tag= 1)
	#communicate
	#send & recv
	for i in range (3):
		COMM.barrier()
		if i==my_pe_num:
			MPI.COMM_WORLD.Send(master_last[i][250,:], dest=i+1, tag=1)
		if i+1==my_pe_num:
			MPI.COMM_WORLD.Send(master_last[i+1][1,:], dest=i, tag=1)
	for i in range (3):
		COMM.barrier()
		if i ==my_pe_num:
			MPI.COMM_WORLD.Recv(master_last[i][251,:], source = i+1, tag= 1)
		if i+1==my_pe_num:
			MPI.COMM_WORLD.Recv(master_last[i+1][0,:], source = i, tag= 1)
	print(iteration, dt_final)
	iteration=iteration+1



master_piece=master_last[my_pe_num]
master_final=COMM.gather(master_piece, root=0)
master_final=np.concatenate( master_final, axis=0)
#delete ghost cells
master_final=np.delete(master_final,[251,252,503,504,755,756], axis=0)
#print(master_final)
output(master_final)
end_time=time.time()
print("Total execution time: {} seconds".format(end_time - start_time))
