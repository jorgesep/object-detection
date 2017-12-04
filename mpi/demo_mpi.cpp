//#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
//#include <omp.h>
#include <vector>
#include <math.h>
#include <stddef.h>

#define send_data_tag 2001
#define return_data_tag 2002

// Create special struct to encapsulate data
typedef struct res_type {
  int id;
  double prob;
} result_type;


int main(int argc, char *argv[]) {
  int numprocs, rank;
  int iam = 0, np = 1;

  // Declaration of MPI variables 
  int procs_id, name_len,
      root_process=0, ierr, i, 
      num_rows, num_procs,idx, 
      num_rows_to_receive, avg_rows_per_process, 
      mod_rows_per_process,
      sender, num_rows_received, 
      start_row, end_row, num_rows_to_send;
  long int sum, partial_sum;

  // initialize an array 
  num_rows =24;
  std::vector<int> dummy2(num_rows);

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Get the number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // Get the rank of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &procs_id);

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(processor_name, &name_len);

  MPI_Status status;
  MPI_Datatype mpi_result_type;
  int blen[2]               = {1,1};
  MPI_Datatype types[2]     = {MPI_INT, MPI_DOUBLE};
  //MPI_Aint displacements[2] = { (size_t)&(result_type.id)   - (size_t)&result_type,
  //                              (size_t)&(result_type.prob) - (size_t)&result_type } ;

  MPI_Aint displacements[2] = { offsetof(result_type, id), offsetof(result_type, prob) };

  MPI_Type_struct( 2, blen, displacements, types, &mpi_result_type );
  MPI_Type_commit(&mpi_result_type);
 
  avg_rows_per_process = (int)ceil((double)num_rows /(double)num_procs);
  mod_rows_per_process = num_rows % num_procs;

  std::cout << "Process name: " << processor_name << " process id: " << procs_id << "/" << num_procs 
            << " avg_rows:" << avg_rows_per_process << "|" << mod_rows_per_process << std::endl;

  if(procs_id == root_process) {

  //// print all command line arguments
  //std::cout << processor_name << " "  << procs_id << " "
  //          << "number of arguments: " << argc << " "
  //          << "name of program: "    << argv[0] ;
  //if (argc > 2)
  //  std::cout << " arg1 : " << argv[1] << " arg2: " << argv[2] << '\n' ;
  //else std::cout << '\n' ;


    std::vector<int> dummy;
    for(int i=0; i<num_rows; i++)
      dummy.push_back(i + 1);
 
    for(int i=0; i<num_rows; i++)
      std::cout << i << "/" << dummy[i] << " ";
    std::cout << std::endl << std::endl;

    /* distribute a portion of the vector to each child process */
    for (idx = 1; idx < num_procs; idx++) {

        start_row = idx*avg_rows_per_process + 1;
        end_row   = (idx + 1)*avg_rows_per_process;


      if ((num_rows - end_row) < avg_rows_per_process)
        end_row = num_rows - 1;
      num_rows_to_send = end_row - start_row + 1;

      std::cout << "PROCESS_" << procs_id << ">>> Sending " 
                << num_rows_to_send << "/"<< num_rows 
                << " rows to process: " << idx 
                << " from: " << processor_name << " [" << start_row << ":" << end_row << "] ... " ;

      ierr = MPI_Send( &num_rows_to_send, 1 , MPI_INT, idx, send_data_tag, MPI_COMM_WORLD);

      ierr = MPI_Send( &dummy[start_row], num_rows_to_send, MPI_INT, idx, send_data_tag, MPI_COMM_WORLD);

      std::cout << "Sent to process " << idx << " " ;
      for(int k=start_row; k < start_row + num_rows_to_send; k++)
      //for(int k=start_row; k < end_row; k++)
        std::cout << "d["<< k << "]=" << dummy[k] << " " ;
      std::cout << std::endl ;

    }

    ///* and calculate the sum of the values in the segment assigned
    // * to the root process */
    sum = 0;
    for (int j = 0; j < avg_rows_per_process + 1; j++){
      sum += dummy[j];

      std::cout << "d[" << j << "]=" << dummy[j] << " " ;
    }

    std::cout << "sum " << sum << " calculated by root process" << std::endl ;


    ///* and, finally, I collet the partial sums from the slave processes,
    // * print them, and add them to the grand sum, and print it */
    for(idx = 1; idx < num_procs; idx++) {

      result_type recv;
      ierr = MPI_Recv( &recv, 1, mpi_result_type, MPI_ANY_SOURCE, return_data_tag, MPI_COMM_WORLD, &status);

      sum += recv.prob;
      std::cout << "PROCESS_" << procs_id << "<< index_" << idx << "/" << num_procs << " "
                << "Partial sum " << recv.prob << "/" << sum << " returned from process " << sender << std::endl ;

      //ierr = MPI_Recv( &partial_sum, 1, MPI_LONG, MPI_ANY_SOURCE, return_data_tag, MPI_COMM_WORLD, &status);
      //sender = status.MPI_SOURCE;
      //sum += partial_sum;
      //std::cout << "PROCESS_" << procs_id << "<< index_" << idx << "/" << num_procs << " "
      //          << "Partial sum " << partial_sum << "/" << sum << " returned from process " << sender << std::endl ;
    }

    std::cout << "The grand total is: " << sum  << std::endl ;

  }

  else {
    ///* I must be a slave process, so I must receive my array segment,
    // * storing it in a "local" array, array1. */
    ierr = MPI_Recv( &num_rows_to_receive, 1, MPI_INT, root_process, send_data_tag, MPI_COMM_WORLD, &status);

    // make space for ints num_rows_to_receive
    dummy2.resize(num_rows_to_receive);
    ierr = MPI_Recv( &dummy2[0], num_rows_to_receive, MPI_INT, root_process, send_data_tag, MPI_COMM_WORLD, &status);

    num_rows_received = num_rows_to_receive;

    ///* Calculate the sum of my portion of the array */
    partial_sum = 0;
    for(i = 0; i < num_rows_received; i++)
      partial_sum += dummy2[i];

    std::cout << "PROCESS_" << procs_id << "<< received= " << num_rows_to_receive << "/" << num_rows << " rows partial sum= " << partial_sum << std::endl ;

    /* and finally, send my partial sum to hte root process */
    //ierr = MPI_Send( &partial_sum, 1, MPI_LONG, root_process, return_data_tag, MPI_COMM_WORLD);

    result_type send;
    send.prob = partial_sum;
    ierr = MPI_Send( &send, 1, mpi_result_type, root_process, return_data_tag, MPI_COMM_WORLD);
  }






  MPI_Type_free(&mpi_result_type);
  // Finalize the MPI environment.
  MPI_Finalize();

}
