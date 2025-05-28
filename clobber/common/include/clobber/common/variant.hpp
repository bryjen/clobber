#pragma once

template <class... Fs> struct overload : Fs... {
    using Fs::operator()...;
};

template <class... Fs> overload(Fs...) -> overload<Fs...>;