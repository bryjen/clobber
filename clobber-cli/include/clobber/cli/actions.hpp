#ifndef ACTIONS_HPP
#define ACTIONS_HPP

struct NewArgs;
struct BuildArgs;
struct RunArgs;

int _new(const NewArgs &); // underscore to prevent name collision with keyword 'new'

int build(const BuildArgs &);

int run(const RunArgs &);

#endif // ACTIONS_HPP