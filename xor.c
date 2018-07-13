#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
//num of units
#define NUM_INPUT 2
#define NUM_HIDDEN 20
#define NUM_TRAIN 4

double sigmoid(double x) {
  return 1/(1+exp(-x));
}

//derivative of sigmoid function
double d_sigmoid(double x) {
  double a = 0.1;
  return a*x*(1-x);
}

int main(void) {
  srand((unsigned)time(NULL));
  //train data
  double train_x[NUM_TRAIN][NUM_INPUT+1] = {{0, 0, -1},{0, 1, -1},{1, 0, -1},{1, 1, -1}};
  double production_X[4][NUM_INPUT+1] = {{0,0,-1},{1,1,-1},{0,1,-1},{1,0,-1}};
  double d[NUM_TRAIN] = {0, 1, 1, 0};
  //net
  double w[NUM_HIDDEN+1][NUM_INPUT+1];
  double v[NUM_HIDDEN+1];
  double y[NUM_TRAIN][NUM_HIDDEN+1];
  double z[NUM_TRAIN];
  double eta = 0.1;
  int epoch = 1000000;
  //other
  int i, j, k, l;
  double tmp = 0;

  //update weights using rand()
  for(l=0; l<NUM_HIDDEN+1; l++) {
    for(i=0; i<NUM_INPUT+1; i++) {
      w[l][i] = ((double)rand() / ((double)RAND_MAX + 1));
    }
  }
  for(i=0; i<NUM_HIDDEN+1; i++) {
    v[i] = ((double)rand() / ((double)RAND_MAX + 1));
  }

  //tain
  for(k=0; k<epoch; k++) {
    //feedforward
    for(j=0; j<NUM_TRAIN; j++) {
      //hidden
      for(l=0; l<NUM_HIDDEN; l++) {
        for(i=0; i<NUM_INPUT+1; i++) {
          tmp += train_x[j][i] * w[l][i];
        }
        y[j][l] = sigmoid(tmp);
        tmp = 0;
      }
      y[j][NUM_HIDDEN] = -1;
      //output
      for(i=0; i<NUM_HIDDEN+1; i++) {
        tmp += y[j][i] * v[i];
      }
      z[j] = sigmoid(tmp);
      tmp = 0;

      //backward
      //output
      for(i=0; i<NUM_HIDDEN+1; i++) {
        v[i] = v[i] - eta * y[j][i] * d_sigmoid(z[j]) * (z[j] - d[j]);
      }

      //hidden
      for(l=0; l<NUM_INPUT+1; l++) {
        for(i=0; i<NUM_HIDDEN+1; i++) {
          w[i][l] = w[i][l] - eta * train_x[j][l] * d_sigmoid(y[j][i]) * d_sigmoid(z[j]) * (z[j] - d[j]) * v[i];
        }
      }
    }
    //print detail
    //  printf("z=");
    for(i=0; i<NUM_TRAIN; i++) {
      //      printf("%f ", z[i]);
    }
    printf("los:%f",(z[j] - d[j])*(z[j] - d[j])/2);
    printf("epoch:%d\n",k);
  }

  //predict
  for(j=0; j<NUM_TRAIN; j++) {
    //hidden
    for(l=0; l<NUM_HIDDEN; l++) {
      for(i=0; i<NUM_INPUT+1; i++) {
        tmp += production_X[j][i] * w[l][i];
      }
      y[j][l] = sigmoid(tmp);
      tmp = 0;
    }
    y[j][NUM_HIDDEN] = -1;
    //output
    for(i=0; i<NUM_HIDDEN+1; i++) {
      tmp += y[j][i] * v[i];
    }
    z[j] = sigmoid(tmp);
    tmp = 0;
  }

  //print result
  printf("z=");
  for(i=0; i<NUM_TRAIN; i++) {
    printf("%f ", z[i]);
  }
  printf("epoch:%d\n",k);
  return 0;
}
