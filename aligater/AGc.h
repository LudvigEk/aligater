void c_Stat_GetMeanAndVariance_double(const double* aData, const int nSize, double &mean, double &var);
void Stat_GetSums_double(const double *aData, const int nSize, double &s, double &ss);
void c_multiply (double* array, double value, int &m, int &n);

inline double Stat_GetVariance(const double s, const double ssqr, const int nSize)
{	
	if (nSize<=1) 
		return 0; 
	else
		return (ssqr-s*s/nSize)/(nSize-1); 
}
