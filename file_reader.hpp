#include <boost/fusion/adapted.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

namespace qi = boost::spirit::qi;

struct double6
{
    double x, y, z, vx, vy, vz;
};

BOOST_FUSION_ADAPT_STRUCT(double6, (double, x)(double, y)(double, z)(double, vx)(double, vy)(double, vz))

typedef std::vector<double6> data_t;

void fileReaderFast(std::string filename, data_t &data)
{
    boost::iostreams::mapped_file mmap(
        filename,
        boost::iostreams::mapped_file::readonly);
    auto f = mmap.const_data();
    auto l = f + mmap.size();

    using namespace qi;
    bool ok = phrase_parse(f, l, (double_ > double_ > double_ > double_ > double_ > double_) % eol, blank, data);
    if (ok)
        std::cout << "parse success\n";
    else
        std::cerr << "parse failed: '" << std::string(f, l) << "'\n";

    if (f != l)
        std::cerr << "trailing unparsed: '" << std::string(f, l) << "'\n";
}
