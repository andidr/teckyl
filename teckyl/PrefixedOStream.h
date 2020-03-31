#ifndef TECKYL_PREFIXED_OSTREAM_H
#define TECKYL_PREFIXED_OSTREAM_H

#include <iostream>

namespace teckyl {

// A stream buffer that outputs a prefix string after each occurence
// of `nlc`, except after the last occurrence (e.g., after each
// newline character except after the last newline character if `nlc`
// is '\n') and optionally at the very beginning.
template <class CharT, class Traits = std::char_traits<CharT>, CharT nlc = '\n'>
class BasicPrefixedStreamBuffer : public std::basic_streambuf<CharT, Traits> {
protected:
  const std::basic_string<CharT, Traits> prefix;
  std::basic_streambuf<CharT, Traits> *buffer;
  bool needPrefix;

  int sync() { return buffer->pubsync(); }

  int overflow(int c) {
    if (c != Traits::eof()) {
      if (needPrefix && !prefix.empty() &&
          prefix.size() != buffer->sputn(prefix.data(), prefix.size())) {
        return Traits::eof();
      }

      needPrefix = (c == nlc);
    }

    return buffer->sputc(c);
  }

public:
  BasicPrefixedStreamBuffer(const std::basic_string<CharT, Traits> &prefix,
                            std::basic_streambuf<CharT, Traits> *buffer,
                            bool prefixFirst = true)
      : prefix(prefix), buffer(buffer), needPrefix(prefixFirst) {}
};

using PrefixedStreamBuffer = BasicPrefixedStreamBuffer<char>;

// An wrapper for an output stream that outputs a prefix after every
// occurence of `nlc`, except after the last occurrence (e.g., after
// each newline character except after the last newline character if
// `nlc` is '\n') and optionally at the very beginning.
template <class CharT, class Traits = std::char_traits<CharT>, CharT nlc = '\n'>
class BasicPrefixedOStream : public std::ostream {
protected:
  BasicPrefixedStreamBuffer<CharT, Traits, nlc> psb;

public:
  BasicPrefixedOStream(std::basic_string<CharT, Traits> const &prefix,
                       std::basic_ostream<CharT, Traits> &out,
                       bool prefixFirst = true)
      : psb(prefix, out.rdbuf(), prefixFirst),
        std::ostream(static_cast<std::basic_streambuf<CharT, Traits> *>(&psb)) {
  }
};

using PrefixedOStream = BasicPrefixedOStream<char>;

} // namespace teckyl

#endif
