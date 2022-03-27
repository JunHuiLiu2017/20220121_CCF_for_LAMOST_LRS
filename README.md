# 20220121_CCF_for_LAMOST_LRS

CCF_test_no_labal.py is the program focus on the true spectral file, i.e., XXX.fits.gz

#########################################################################################
0. Generate the CCF test sample

   (0.1) The materials used by the <Generate_CCF_test_sample.py> program are 
the whole sample (<2022_01_03_13_54_24_74768Apdr16_para_dump_ap_lm.dpl>) and 
the filter 1, 2 and 3 files.
These filters contain the necessary filer information to remove the bad spectra 
that can not be predicted by CCF model (In fact, they are the objects which 
can not be predicted well by slam model).

   (0.2) Then utilize the <Generate_CCF_test_sample.py> to generate the test sample 
that be compassed in dump file <small_sample_3000.dump>. For this <small_sample_3000.dump>,
it contains the spectra file in mrs form and the parameters from APOGEE and LAMOST for the corresponding spectra.

##########################################################################################
1. Do the CCF test
   Files: small_sample_3000.dump (test sample); CCF_test_with_label.py; 2022_03_16_14_08_52_imitated_CCF_wl_3800_6000_1000_.dump (CCF model spectra)
   CAUTION: In the pre-process, the negative flux values are neglected,
    then their original flux values are interpolated as positive values.
    Meanwhile, for the negative norm flux they are set as 1.


##########################################################################################
2. Do the CCF

   (1.1) The materials used by this CCF program are 
the fits files (like folder <20220313_CCF_test_grouped_fits>)
, the data record in "csv" form (like <20220313_snrg_30_Grpeddata_CCF_test.csv>)
and the CCF model spectra (like <2022_03_16_14_08_52_imitated_CCF_wl_3800_6000_1000_.dump>).

   (1.2) Put the paths of the fits files, data record and CCF model spectra into <CCF_test_no_label.py>

   (1.3) Then the results with be stored in ".dump" form.

#########################################################################################
2. Process the CCF Results 

(2.1) This function are written in Process_CCF_results.py.

(2.2) Read the data recording files in folder CCF_results, then rewrite <.dump> to <CCF_result.csv>.


#########################################################################################














