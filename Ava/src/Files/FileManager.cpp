#include <avapch.h>
#include "FileManager.h"

#include <Files/File.h>
#include <Files/FilePath.h>
#include <Time/Profiler.h>
#include <Events/FileEvent.h>
#include <Debug/Assert.h>
#include <Math/Hash.h>

namespace Ava {

    // ----- File manager lifecycle -----------------------------------------------------------

    FileMgr* FileMgr::s_instance = nullptr;

    FileMgr::FileMgr()
    {
    }

    FileMgr::~FileMgr()
    {
    }

    void FileMgr::Init()
    {
        if (!s_instance)
        {
            s_instance = new FileMgr();
        }
    }

    void FileMgr::Shutdown()
    {
        if (s_instance)
        {
            delete s_instance;
        }
    }

}
