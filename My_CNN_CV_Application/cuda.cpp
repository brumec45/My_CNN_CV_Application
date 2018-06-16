#include "cuda.h"

void checkCUDNN(cudnnStatus_t status)                               
  {                                                          
      if (status != CUDNN_STATUS_SUCCESS) {                    
      std::cout << "Error on line " << __LINE__ << ": "      
                << cudnnGetErrorString(status) << std::endl; 
      std::exit(EXIT_FAILURE);        
	  
    }
	else 
	{
		std::cout << "OK" << std::endl; 
	}
  }