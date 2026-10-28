// Files, paths and the socket wire for the jobserver. The wire is one JSON
// document per line: the client sends one request line, the server answers one
// line and closes (a `wait` request is answered when the job finishes).
#pragma once

#include "jobs/json.hpp"

#include <expected>
#include <string>

namespace jobs {

std::expected<std::string, std::string> readFile(const std::string& path);
// write to path.tmp, fsync, rename: a reader never sees a torn record.
std::expected<void, std::string> writeFileAtomic(const std::string& path, const std::string& data);

// $SIMJOBS_SOCKET, else $XDG_RUNTIME_DIR/simjobs.sock, else /tmp/simjobs-<uid>.sock.
std::string defaultSocketPath();
// $XDG_STATE_HOME/simjobs, else ~/.local/state/simjobs.
std::string defaultStateDir();

double nowUnix();

// One request/response round trip. Fails on connect/IO/parse errors; a
// server-side refusal comes back as {"ok": false, "error": ...}.
std::expected<json::Value, std::string> request(const std::string& socketPath, const json::Value& req);

} // namespace jobs
