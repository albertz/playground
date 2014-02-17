
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

// this needs to link in boost
//#include <boost/thread/mutex.hpp>

int main() {
	boost::shared_ptr<int> x;
	boost::weak_ptr<int> y(x);

	//boost::mutex m;
}

