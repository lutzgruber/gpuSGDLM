#ifdef USE_RCPP

#include "SGDLM/RcppWrapper.hpp"

#include "SGDLM/HostWrapperFactory.hpp"

#include "logger.hpp"

template<typename DOUBLE> SGDLM::RcppWrapper<DOUBLE>::RcppWrapper() :
		wrapper(SGDLM::HostWrapperFactory<DOUBLE>::create(1)) {
	DEBUG_LOGGER << "RcppWrapper::RcppWrapper()" << ENDL;
}

template<typename DOUBLE> SGDLM::RcppWrapper<DOUBLE>::RcppWrapper(std::size_t no_gpus) :
		wrapper(SGDLM::HostWrapperFactory<DOUBLE>::create(no_gpus)) {
	DEBUG_LOGGER << "RcppWrapper::RcppWrapper(" << no_gpus << ")" << ENDL;
}

template<typename DOUBLE> SGDLM::RcppWrapper<DOUBLE>::~RcppWrapper() {
	DEBUG_LOGGER << "RcppWrapper::~RcppWrapper()" << ENDL;

	wrapper->clearMemory();

	delete wrapper;
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::clearGPUAllMemory() {
	DEBUG_LOGGER << "RcppWrapper::clearGPUAllMemory()" << ENDL;

	wrapper->clearMemory();

	//this->host_MEM.clear();
	//this->host_MEM_temp.clear();
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::clearGPUSimMemory() {
	DEBUG_LOGGER << "RcppWrapper::clearGPUSimMemory()" << ENDL;

	wrapper->clearSimMemory();
}

template<typename DOUBLE> Rcpp::List SGDLM::RcppWrapper<DOUBLE>::getSimultaneousParents() const {
	DEBUG_LOGGER << "RcppWrapper::getSimultaneousParents()" << ENDL;

	std::size_t m = wrapper->getNoSeries();
	std::size_t max_p = wrapper->getMaxP();

	std::size_t dim[] = { max_p, m };
	std::size_t no_elements = dim[0] * dim[1];

	memory_manager_RCPP host_MEM_temp;

	unsigned int* data_p = host_MEM_temp.host_alloc_vec<unsigned int>(m);
	unsigned int* data_sp = host_MEM_temp.host_alloc_vec<unsigned int>(no_elements);

	wrapper->getParentalSets(data_p, data_sp);

	std::vector<unsigned int> p = memory_manager_RCPP::getSTDVector<unsigned int>(data_p, m);
	Rcpp::IntegerVector sp = memory_manager_RCPP::getRcppVector<unsigned int, Rcpp::IntegerVector>(data_sp, 2, dim);

	return Rcpp::List::create(Rcpp::Named("p") = p, Rcpp::Named("sp") = sp);
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::setSimultaneousParents(const std::vector<unsigned int>& p,
		const Rcpp::IntegerVector& sp) {
	DEBUG_LOGGER << "RcppWrapper::setSimultaneousParents()" << ENDL;

	Rcpp::Dimension dim_arg_2 = sp.attr("dim");

	if (dim_arg_2.size() != 2) {
		ERROR_LOGGER << "Incorrect dimension of sp" << ENDL;
		return;
	}

	std::size_t m_arg_1 = p.size();
	std::size_t m_arg_2 = dim_arg_2[1];
	std::size_t max_p_arg_2 = dim_arg_2[0];

	if (m_arg_1 != m_arg_2) {
		ERROR_LOGGER << "Dimensions of input arguments are incompatible" << ENDL;
		return;
	}

	memory_manager_RCPP host_MEM_temp;

	unsigned int* data_p = host_MEM_temp.getData<unsigned int, std::vector<unsigned int> >(p);
	unsigned int* data_sp = host_MEM_temp.getData<unsigned int, Rcpp::IntegerVector>(sp);

	DEBUG_LOGGER << "data_p[" << m_arg_1 << "]:";
	unsigned int* running_pointer = data_p;
	for (std::size_t i = 0; i < m_arg_1; i++, running_pointer++) {
		DEBUG_LOGGER << " [" << data_p[i] << ", " << *running_pointer << "]";
	}
	DEBUG_LOGGER << ENDL;

	DEBUG_LOGGER << "data_sp[" << (m_arg_2 * max_p_arg_2) << "]:";
	running_pointer = data_sp;
	for (std::size_t i = 0; i < m_arg_2 * max_p_arg_2; i++, running_pointer++) {
		DEBUG_LOGGER << " [" << data_sp[i] << ", " << *running_pointer << "]";
	}
	DEBUG_LOGGER << ENDL;

	if (wrapper->getNoSeries() != m_arg_1 || wrapper->getMaxP() != max_p_arg_2) {
		// must re-initialize device memory
		wrapper->initMemory(m_arg_1, max_p_arg_2);
	}

	wrapper->setParentalSets(data_p, data_sp);
}

template<typename DOUBLE> Rcpp::List SGDLM::RcppWrapper<DOUBLE>::getDiscountFactors() const {
	DEBUG_LOGGER << "RcppWrapper::getDiscountFactors()" << ENDL;

	std::size_t m = wrapper->getNoSeries();
	std::size_t max_p = wrapper->getMaxP();

	std::size_t dim[] = { max_p, max_p, m };
	std::size_t no_elements = dim[0] * dim[1] * dim[2];

	memory_manager_RCPP host_MEM_temp;

	DOUBLE* data_beta = host_MEM_temp.host_alloc_vec<DOUBLE>(m);
	DOUBLE* data_delta = host_MEM_temp.host_alloc_vec<DOUBLE>(no_elements);

	wrapper->getDiscountFactors(data_beta, data_delta);

	std::vector<DOUBLE> beta = memory_manager_RCPP::getSTDVector<DOUBLE>(data_beta, m);
	Rcpp::NumericVector delta = memory_manager_RCPP::getRcppVector<DOUBLE, Rcpp::NumericVector>(data_delta, 3, dim);

	return Rcpp::List::create(Rcpp::Named("beta") = beta, Rcpp::Named("delta") = delta);
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::setDiscountFactors(const std::vector<DOUBLE>& beta,
		const Rcpp::NumericVector& delta) {
	DEBUG_LOGGER << "RcppWrapper::setDiscountFactors()" << ENDL;

	Rcpp::Dimension dim_arg_2 = delta.attr("dim");

	if (dim_arg_2.size() != 3) {
		ERROR_LOGGER << "Incorrect dimension of delta" << ENDL;
		return;
	}

	std::size_t m = wrapper->getNoSeries();
	std::size_t max_p = wrapper->getMaxP();

	std::size_t m_arg_1 = beta.size();
	std::size_t m_arg_2 = dim_arg_2[2];
	std::size_t max_p_arg_2_0 = dim_arg_2[0];
	std::size_t max_p_arg_2_1 = dim_arg_2[1];

	if (m_arg_1 != m || m_arg_2 != m || max_p_arg_2_0 != max_p || max_p_arg_2_1 != max_p) {
		ERROR_LOGGER << "Dimensions of input arguments are incompatible" << ENDL;
		return;
	}

	memory_manager_RCPP host_MEM_temp;

	DOUBLE* data_beta = host_MEM_temp.getData<DOUBLE, std::vector<DOUBLE> >(beta);
	DOUBLE* data_delta = host_MEM_temp.getData<DOUBLE, Rcpp::NumericVector>(delta);

	wrapper->setDiscountFactors(data_beta, data_delta);
}

template<typename DOUBLE> Rcpp::List SGDLM::RcppWrapper<DOUBLE>::getParameters() const {
	DEBUG_LOGGER << "RcppWrapper::getParameters()" << ENDL;

	memory_manager_RCPP host_MEM_temp;

	DOUBLE* data_m_t = host_MEM_temp.host_alloc_vec<DOUBLE>(wrapper->getNoSeries() * wrapper->getMaxP());
	DOUBLE* data_C_t = host_MEM_temp.host_alloc_vec<DOUBLE>(
			wrapper->getNoSeries() * wrapper->getMaxP() * wrapper->getMaxP());
	DOUBLE* data_n_t = host_MEM_temp.host_alloc_vec<DOUBLE>(wrapper->getNoSeries());
	DOUBLE* data_s_t = host_MEM_temp.host_alloc_vec<DOUBLE>(wrapper->getNoSeries());

	wrapper->getParameters(data_m_t, data_C_t, data_n_t, data_s_t);

	std::size_t dim_m_t[] = { wrapper->getMaxP(), wrapper->getNoSeries() };
	std::size_t dim_C_t[] = { wrapper->getMaxP(), wrapper->getMaxP(), wrapper->getNoSeries() };

	Rcpp::NumericVector m_t = memory_manager_RCPP::getRcppVector<DOUBLE, Rcpp::NumericVector>(data_m_t, 2, dim_m_t);
	Rcpp::NumericVector C_t = memory_manager_RCPP::getRcppVector<DOUBLE, Rcpp::NumericVector>(data_C_t, 3, dim_C_t);
	std::vector<DOUBLE> n_t = memory_manager_RCPP::getSTDVector<DOUBLE>(data_n_t, wrapper->getNoSeries());
	std::vector<DOUBLE> s_t = memory_manager_RCPP::getSTDVector<DOUBLE>(data_s_t, wrapper->getNoSeries());

	return Rcpp::List::create(Rcpp::Named("m") = m_t, Rcpp::Named("C") = C_t, Rcpp::Named("n") = n_t, Rcpp::Named("s") =
			s_t);
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::setPriorParameters(const Rcpp::NumericVector& m_t,
		const Rcpp::NumericVector& C_t, const std::vector<DOUBLE>& n_t, const std::vector<DOUBLE>& s_t) {
	DEBUG_LOGGER << "RcppWrapper::setPriorParameters()" << ENDL;

	setParameters(m_t, C_t, n_t, s_t, true);
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::setPosteriorParameters(const Rcpp::NumericVector& m_t,
		const Rcpp::NumericVector& C_t, const std::vector<DOUBLE>& n_t, const std::vector<DOUBLE>& s_t) {
	DEBUG_LOGGER << "RcppWrapper::setPosteriorParameters()" << ENDL;

	setParameters(m_t, C_t, n_t, s_t, false);
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::setParameters(const Rcpp::NumericVector& m_t,
		const Rcpp::NumericVector& C_t, const std::vector<DOUBLE>& n_t, const std::vector<DOUBLE>& s_t, bool is_prior) {
	DEBUG_LOGGER << "RcppWrapper::setParameters()" << ENDL;

	Rcpp::Dimension dim_arg_1 = m_t.attr("dim");
	Rcpp::Dimension dim_arg_2 = C_t.attr("dim");

	if (dim_arg_1.size() != 2 || dim_arg_2.size() != 3) {
		ERROR_LOGGER << "Incorrect dimension of the input arguments" << ENDL;
		return;
	}

	std::size_t m_arg_1 = dim_arg_1[1];
	std::size_t m_arg_2 = dim_arg_2[2];
	std::size_t m_arg_3 = n_t.size();
	std::size_t m_arg_4 = s_t.size();

	std::size_t max_p_arg_1 = dim_arg_1[0];
	std::size_t max_p_arg_2_1 = dim_arg_2[0];
	std::size_t max_p_arg_2_2 = dim_arg_2[1];

	if (m_arg_1 != wrapper->getNoSeries() || m_arg_2 != wrapper->getNoSeries() || m_arg_3 != wrapper->getNoSeries()
			|| m_arg_4 != wrapper->getNoSeries() || max_p_arg_1 != wrapper->getMaxP()
			|| max_p_arg_2_1 != wrapper->getMaxP() || max_p_arg_2_2 != wrapper->getMaxP()) {
		ERROR_LOGGER << "Dimensions of input arguments are incompatible" << ENDL;
		return;
	}

	memory_manager_RCPP host_MEM_temp;

	DOUBLE* data_m_t = host_MEM_temp.getData<DOUBLE, Rcpp::NumericVector>(m_t);
	DOUBLE* data_C_t = host_MEM_temp.getData<DOUBLE, Rcpp::NumericVector>(C_t);
	DOUBLE* data_n_t = host_MEM_temp.getData<DOUBLE, std::vector<DOUBLE> >(n_t);
	DOUBLE* data_s_t = host_MEM_temp.getData<DOUBLE, std::vector<DOUBLE> >(s_t);

	wrapper->setParameters(data_m_t, data_C_t, data_n_t, data_s_t);

	wrapper->isPrior(is_prior);
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::computePosterior(const std::vector<DOUBLE>& y_t,
		const Rcpp::NumericVector& F_t) {
	DEBUG_LOGGER << "RcppWrapper::computePosterior()" << ENDL;

	Rcpp::Dimension dim_arg_2 = F_t.attr("dim");

	if (dim_arg_2.size() != 2) {
		ERROR_LOGGER << "Incorrect dimension of the input arguments" << ENDL;
		return;
	}

	std::size_t m_arg_1 = y_t.size();
	std::size_t m_arg_2 = dim_arg_2[1];

	std::size_t max_p_arg_2 = dim_arg_2[0];

	if (m_arg_1 != wrapper->getNoSeries() || m_arg_2 != wrapper->getNoSeries() || max_p_arg_2 != wrapper->getMaxP()) {
		ERROR_LOGGER << "Dimensions of input arguments are incompatible" << ENDL;
		return;
	}

	memory_manager_RCPP host_MEM_temp;

	DOUBLE* data_y_t = host_MEM_temp.getData<DOUBLE, std::vector<DOUBLE> >(y_t);
	DOUBLE* data_F_t = host_MEM_temp.getData<DOUBLE, Rcpp::NumericVector>(F_t);

	wrapper->computePosterior(data_y_t, data_F_t);
}

template<typename DOUBLE> Rcpp::List SGDLM::RcppWrapper<DOUBLE>::computeVBPosterior(std::size_t nsim,
		std::size_t nsim_batch) {
	DEBUG_LOGGER << "RcppWrapper::computeVBPosterior()" << ENDL;

	if (nsim != wrapper->getNSim() || nsim_batch != wrapper->getNSimBatch()) { // initialize simulation memory
		wrapper->initSimMemory(nsim, nsim_batch);
	}

	memory_manager_RCPP host_MEM_temp;

	// allocate host output memory
	DOUBLE* data_mean_m_t = host_MEM_temp.host_alloc_vec<DOUBLE>(wrapper->getNoSeries() * wrapper->getMaxP());
	DOUBLE* data_mean_C_t = host_MEM_temp.host_alloc_vec<DOUBLE>(
			wrapper->getNoSeries() * wrapper->getMaxP() * wrapper->getMaxP());
	DOUBLE* data_mean_n_t = host_MEM_temp.host_alloc_vec<DOUBLE>(wrapper->getNoSeries());
	DOUBLE* data_mean_s_t = host_MEM_temp.host_alloc_vec<DOUBLE>(wrapper->getNoSeries());
	DOUBLE* data_IS_weights = host_MEM_temp.host_alloc_vec<DOUBLE>(wrapper->getNSim());
	DOUBLE sum_det_weights = 0;

	// compute VB posterior on GPU
	wrapper->computeVBPosterior(data_mean_m_t, data_mean_C_t, data_mean_n_t, data_mean_s_t, data_IS_weights,
			&sum_det_weights);

	// set up return dimensions
	std::size_t dim_m_t[] = { wrapper->getMaxP(), wrapper->getNoSeries() };
	std::size_t dim_C_t[] = { wrapper->getMaxP(), wrapper->getMaxP(), wrapper->getNoSeries() };

	// create R return values
	Rcpp::NumericVector m_t = memory_manager_RCPP::getRcppVector<DOUBLE, Rcpp::NumericVector>(data_mean_m_t, 2,
			dim_m_t);
	Rcpp::NumericVector C_t = memory_manager_RCPP::getRcppVector<DOUBLE, Rcpp::NumericVector>(data_mean_C_t, 3,
			dim_C_t);
	std::vector<DOUBLE> n_t = memory_manager_RCPP::getSTDVector<DOUBLE>(data_mean_n_t, wrapper->getNoSeries());
	std::vector<DOUBLE> s_t = memory_manager_RCPP::getSTDVector<DOUBLE>(data_mean_s_t, wrapper->getNoSeries());
	std::vector<DOUBLE> IS_weights = memory_manager_RCPP::getSTDVector<DOUBLE>(data_IS_weights, wrapper->getNSim());

	return Rcpp::List::create(Rcpp::Named("m") = m_t, Rcpp::Named("C") = C_t, Rcpp::Named("n") = n_t, Rcpp::Named("s") =
			s_t, Rcpp::Named("IS_weights") = IS_weights, Rcpp::Named("sum_det_weights") = sum_det_weights);
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::computePrior() {
	DEBUG_LOGGER << "RcppWrapper::computePrior()" << ENDL;

	wrapper->computePrior();
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::computeEvoPrior(const Rcpp::NumericVector& G_tp1) {
	DEBUG_LOGGER << "RcppWrapper::computePrior(evo)" << ENDL;

	Rcpp::Dimension dim_arg_1 = G_tp1.attr("dim");

	if (dim_arg_1.size() != 3) {
		ERROR_LOGGER << "Incorrect dimension of the input arguments" << ENDL;
		return;
	}

	std::size_t m_arg_1 = dim_arg_1[2];

	std::size_t max_p_arg_1_1 = dim_arg_1[0];
	std::size_t max_p_arg_1_2 = dim_arg_1[1];

	if (m_arg_1 != wrapper->getNoSeries() || max_p_arg_1_1 != wrapper->getMaxP()
			|| max_p_arg_1_2 != wrapper->getMaxP()) {
		ERROR_LOGGER << "Dimensions of input arguments are incompatible" << ENDL;
		return;
	}

	memory_manager_RCPP host_MEM_temp;

	DOUBLE* data_G_tp1 = host_MEM_temp.getData<DOUBLE, Rcpp::NumericVector>(G_tp1);

	wrapper->manageEvoMemory(true);

	wrapper->computePrior(data_G_tp1);
}

template<typename DOUBLE> Rcpp::NumericVector SGDLM::RcppWrapper<DOUBLE>::computeForecast(std::size_t nsim,
		std::size_t nsim_batch, const Rcpp::NumericVector& F_tp1) {
	DEBUG_LOGGER << "RcppWrapper::computeForecast()" << ENDL;

	Rcpp::Dimension dim_arg_3 = F_tp1.attr("dim");

	if (dim_arg_3.size() != 2) {
		ERROR_LOGGER << "Incorrect dimension of the input arguments" << ENDL;
		return Rcpp::NumericVector();
	}

	std::size_t m_arg_3 = dim_arg_3[1];

	std::size_t max_p_arg_3 = dim_arg_3[0];

	if (m_arg_3 != wrapper->getNoSeries() || max_p_arg_3 != wrapper->getMaxP()) {
		ERROR_LOGGER << "Dimensions of input arguments are incompatible" << ENDL;
		return Rcpp::NumericVector();
	}

	if (nsim != wrapper->getNSim() || nsim_batch != wrapper->getNSimBatch()) { // initialize simulation memory
		wrapper->initSimMemory(nsim, nsim_batch);
	}

	memory_manager_RCPP host_MEM_temp;

	DOUBLE* data_y_tp1 = host_MEM_temp.host_alloc_vec<DOUBLE>(wrapper->getNSim() * wrapper->getNoSeries());
	DOUBLE* data_F_tp1 = host_MEM_temp.getData<DOUBLE, Rcpp::NumericVector>(F_tp1);

	wrapper->computeForecast(data_y_tp1, data_F_tp1);

	std::size_t dim_y_tp1[] = { wrapper->getNoSeries(), wrapper->getNSim() };

	Rcpp::NumericVector y_tp1 = memory_manager_RCPP::getRcppVector<DOUBLE, Rcpp::NumericVector>(data_y_tp1, 2,
			dim_y_tp1);

	return y_tp1;
}

template<typename DOUBLE> std::size_t SGDLM::RcppWrapper<DOUBLE>::getLogLevel() const {
	DEBUG_LOGGER << "RcppWrapper::getLogLevel()" << ENDL;

	return (std::size_t) LOGACTIVE;
}

template<typename DOUBLE> void SGDLM::RcppWrapper<DOUBLE>::setLogLevel(std::size_t loglevel) {
	DEBUG_LOGGER << "RcppWrapper::setLogLevel(" << loglevel << ")" << ENDL;

	configureLogLevel<std::size_t>(loglevel);
}

/*
 *
 *
 *
 *
 *
 *
 */

// explicit instantiation
template class SGDLM::RcppWrapper<DOUBLETYPE>;

#endif
