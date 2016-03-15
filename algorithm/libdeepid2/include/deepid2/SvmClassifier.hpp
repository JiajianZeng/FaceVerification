#ifndef SVM_CLASSIFIER_HPP
#define SVM_CLASSIFIER_HPP

class SvmClassifier {
 private:
  int nr_class_;              /* number of class, = 2 in regression/one class svm */
  int l_;                     /* total #SV */
  double **sv_coef_;          /* coefficients for SVs in decision functions (sv_coef[k-1][1]) */
  double *rho_;               /* constants in decision functions (rho[k * (k - 1) / 2]) */
  double *probA_;             /* pairwise probability information */ 
  double *probB_;           
  int *sv_indices;            /* sv_indices[0,....,nSV-1] are values in [1,...,num_training_data] to indicates SVs in the training ser */

  /* for classification only */

  int *label;                 /* label of each class (label[k]) */
  int *nSV;                   /* number of SVs for each class (nSV[k]) */
                              /* nSV[0] + ... + nSV[k-1] = 1 */
  
  int free_sv;                /* 1 if svm_model is created by svm_load_model */
                              /* 0 if svm_model is created by svm_train */
}

#endif
