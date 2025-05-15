#pragma once

#include "pch.hpp"

namespace clobber {
    struct Type; // clobber/semantics.hpp
};

std::string type_tostring(const clobber::Type &);