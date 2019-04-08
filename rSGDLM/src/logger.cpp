#include "logger.hpp"


#ifdef USE_MATLAB
#include "mex.h"
#endif

#ifdef USE_RCPP
#include <Rcpp.h>
#endif

LOGLEVEL LOGACTIVE = LOGRESULT;

logstream& logstream::operator<<(const endl& value) {
	if (LOGACTIVE >= loglevel) {
		buffer << "\n";
		std::string msg_str = buffer.str();
		buffer.str("");

#ifdef USE_MATLAB
		const char* msg_char = msg_str.c_str();
		if (loglevel == LOGERROR) {
			mexErrMsgTxt(msg_char);
		} else if (loglevel == LOGWARNING) {
			mexWarnMsgTxt(msg_char);
		} else {
			mexPrintf(msg_char);
		}
#endif

#ifdef USE_RCPP
		if (loglevel == LOGERROR) {
			Rcpp::Rcerr << "Error: " << msg_str << std::endl;
		} else if (loglevel == LOGWARNING) {
			Rcpp::Rcout << "Warning: " << msg_str << std::endl;
		} else {
			Rcpp::Rcout << msg_str << std::endl;
		}
#endif
	}

	return *this;
}

