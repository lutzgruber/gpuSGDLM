/*
 * logger.hpp
 *
 *  Created on: Dec 2, 2013
 *      Author: lutz
 */

#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include <string>
#include <sstream>

//#define LOGGER_LOG_LEVEL_ACTIVE 100

#define LOGGER_LOG_LEVEL_SYSDEBUG  100
#define LOGGER_LOG_LEVEL_DEBUG      50
#define LOGGER_LOG_LEVEL_RESULT     20
#define LOGGER_LOG_LEVEL_INFO       10
#define LOGGER_LOG_LEVEL_WARNING     5
#define LOGGER_LOG_LEVEL_ERROR       0

enum LOGLEVEL {
	LOGSYSDEBUG = 100, LOGDEBUG = 50, LOGRESULT = 20, LOGINFO = 10, LOGWARNING = 5, LOGERROR = 0
};
extern LOGLEVEL LOGACTIVE;

class endl {
public:
	endl() {

	}

	endl(const endl& cpy) {

	}
};
static const endl ENDL;

class logstream {
public:
	logstream(LOGLEVEL level, std::stringstream& buffer) :
			loglevel(level), buffer(buffer) {

	}

	logstream(const logstream& copy) :
			loglevel(copy.loglevel), buffer(copy.buffer) {

	}

	template<typename t> logstream& operator<<(const t& value) {
		if (LOGACTIVE >= loglevel) {
			buffer << value;
		}
		return *this;
	}

	logstream& operator<<(const endl& value);

private:
	LOGLEVEL loglevel;
	std::stringstream& buffer;
};

class NULLstream {
public:
	NULLstream() {
	}

	NULLstream(const NULLstream& copy) {
	}

	template<typename t> const NULLstream& operator<<(const t& value) const {
		return *this;
	}

	const NULLstream& operator<<(const endl& value) const {
		return *this;
	}
};

#if LOGGER_LOG_LEVEL_ACTIVE >= LOGGER_LOG_LEVEL_ERROR
static std::stringstream errstream;
static logstream ERROR_LOGGER(LOGERROR, errstream);
#else
static NULLstream ERROR_LOGGER;
#endif

#if LOGGER_LOG_LEVEL_ACTIVE >= LOGGER_LOG_LEVEL_WARNING
static std::stringstream warningstream;
static logstream WARNING_LOGGER(LOGWARNING, warningstream);
#else
static NULLstream WARNING_LOGGER;
#endif

#if LOGGER_LOG_LEVEL_ACTIVE >= LOGGER_LOG_LEVEL_RESULT
static std::stringstream resultstream;
static logstream RESULT_LOGGER(LOGRESULT, resultstream);
#else
extern NULLstream RESULT_LOGGER;
#endif

#if LOGGER_LOG_LEVEL_ACTIVE >= LOGGER_LOG_LEVEL_INFO
static std::stringstream infostream;
static logstream INFO_LOGGER(LOGINFO, infostream);
#else
static NULLstream INFO_LOGGER;
#endif

#if LOGGER_LOG_LEVEL_ACTIVE >= LOGGER_LOG_LEVEL_DEBUG
static std::stringstream debugstream;
static logstream DEBUG_LOGGER(LOGDEBUG, debugstream);
#else
static NULLstream DEBUG_LOGGER;
#endif

#if LOGGER_LOG_LEVEL_ACTIVE >= LOGGER_LOG_LEVEL_SYSDEBUG
static std::stringstream sysdebugstream;
static logstream SYSDEBUG_LOGGER(LOGSYSDEBUG, sysdebugstream);
#else
static NULLstream SYSDEBUG_LOGGER;
#endif

template<typename NUM> static void configureLogLevel(NUM level) {
	DEBUG_LOGGER << "setLogLevel(" << level << ")" << ENDL;

	if (level >= LOGSYSDEBUG) {
		LOGACTIVE = LOGSYSDEBUG;
		INFO_LOGGER << "New log level: SYSDEBUG" << ENDL;
	} else if (level >= LOGDEBUG) {
		LOGACTIVE = LOGDEBUG;
		INFO_LOGGER << "New log level: DEBUG" << ENDL;
	} else if (level >= LOGRESULT) {
		LOGACTIVE = LOGRESULT;
		INFO_LOGGER << "New log level: RESULT" << ENDL;
	} else if (level >= LOGINFO) {
		LOGACTIVE = LOGINFO;
		INFO_LOGGER << "New log level: INFO" << ENDL;
	} else if (level >= LOGWARNING) {
		LOGACTIVE = LOGWARNING;
		INFO_LOGGER << "New log level: WARNING" << ENDL;
	} else {
		LOGACTIVE = LOGERROR;
		INFO_LOGGER << "New log level: ERROR" << ENDL;
	}
}

/*inline std::string log_dims(std::string name, size_t ndims, const size_t* dims) {
 std::stringstream ss;

 ss << name << " = { ";

 for(size_t i = 0; i < ndims; i++) {
 if(i != 0) {
 ss << ", ";
 }

 ss << dims[i];
 }

 ss << " }" << std::endl;

 return ss.str();
 }*/

#define myAssert(ans) { if(!(ans)) { return; } }

#endif /* LOGGER_HPP_ */
