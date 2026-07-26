// Headless self-check for the widget layer (see guicheck.cpp). Returns the
// number of failures; 0 is a pass. Driven from `dns-ui --check`.
#pragma once

namespace ui {

int guiCheck();

} // namespace ui
