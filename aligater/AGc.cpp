//	______|\__________\o/__________
//			~~aliGater~~
//	(semi)automated gating software
//
//	Björn Nilsson & Ludvig Ekdahl 2016~
//	http://nilssonlab.org
#include "AGc.h"

void c_multiply (double* array, double multiplier, int &m, int &n) {

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



void c_Stat_GetMeanAndVariance_double(const double* aData, const int nSize, double &mean, double &var)
{
	// Special case, small vector
	if (nSize<=1)
	{
		var= 0;
		if (nSize)
			mean= *aData;
		else
			mean= 0;
		return;
	}

	double s, ssqr;
	Stat_GetSums_double(aData, nSize, s, ssqr);

	mean= s/nSize;
	var= Stat_GetVariance(s, ssqr, nSize);
	return;
}

void Stat_GetSums_double(const double *aData, const int nSize, double &s, double &ss)
{
	// Compute sum and sum-of-squares of aData[nSize]
	s= 0;
	ss= 0;
	for (int i=nSize;--i>=0;)
	{
		double tmp= *aData++;
		s += tmp;
		ss += tmp*tmp;
	}
}

