//	______|\__________\o/__________
//			~~aliGater~~
//	(semi)automated gating software
//
//	Björn Nilsson & Ludvig Ekdahl 2016~
//	http://nilssonlab.org

void c_multiply (double* array, double multiplier, int m, int n) {

    int i, j ;
    int index = 0 ;

    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            array[index] = array[index]  * multiplier ;
            index ++ ;
            }
        }
    return ;
}