#include <clocale>
#include <locale>

namespace flatbuffers {
class ClassicLocale {
 public:
  locale_t locale_;
  ClassicLocale() : locale_(newlocale(LC_ALL, "C", nullptr)) {}
  ~ClassicLocale() { freelocale(locale_); }
  static ClassicLocale instance_;
};
ClassicLocale ClassicLocale::instance_;
}

extern "C" int cpuinfo_emscripten_init(void) {
    return 1;
}
