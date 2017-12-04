//#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
//#include <omp.h>

int main(int argc, char *argv[]) {
  int numprocs, rank;
  int iam = 0, np = 1;

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  //MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);


  int source, dest, tag = 0;
  char message[100];
  MPI_Status status;
  if (world_rank != 0) {
    std::cout << "Processor " << processor_name << " rank " << world_rank << " out of " << world_size << " processors" << std::endl;
    sprintf(message,"greetings from process %d!",world_rank);
    dest = 0;
    MPI_Send(message, strlen(message)+1,MPI_CHAR, dest, tag,MPI_COMM_WORLD);
  } else {
    std::cout << "processor 0, " << processor_name << " p = " << world_size << std::endl ;
    for(source=1; source < world_size; source++) {
      MPI_Recv(message,100, MPI_CHAR, source,tag, MPI_COMM_WORLD, &status);
      std::string str(message);
      std::cout << str << std::endl;
    }
  }

  // Finalize the MPI environment.
  MPI_Finalize();

}
