#include <panda/PandaMessageIORequest.h>
#include <panda/PandaMessage.h>
#include <cstring>
#include <mpi.h>

namespace panda
{

  //replace with ADIOS interface
  //write through MPI send and receive

  bool PandaMessage::pollUnsent()
  {
    PandaMessageIOData * data = NULL;
	//TODO
    return true;
  }

  void PandaMessage::pollPending()
  {
    if (pendingIO.empty()) return;
    std::list<PandaMessageIOData * > newPending;
    //TODO
  }//void

  void PandaMessage::pollSends()
  {
    const int MAX_SENDS_PER_LOOP = 20;
    int index = 0;
    while (++index < MAX_SENDS_PER_LOOP && pollUnsent()) { }
    index = 0;
    pollPending();
  }

  
  void PandaMessage::poll(int & finishedWorkers,
                             bool * const workerDone,
                             bool * const recvingCount,
                             int * const counts,
                             int ** keyRecv,
                             int ** valRecv,
                             MPI_Request * recvReqs)
  {
   //TODO
  }//void

  void PandaMessage::grow(const int size, const int finalSize, int & finalSpace, char *& finals)
  {
    if (size + finalSize > finalSpace)
    {
      int newSpace = finalSpace * 2;
      while (size + finalSize > newSpace) newSpace *= 2;
      finalSpace = newSpace;
      char * temp = new char[finalSpace];
      memcpy(temp, finals, finalSize);
      delete [] finals;
      finals = temp;
    }//if
  }//void

  void PandaMessage::privateAdd(const void * const keys, const void * const vals, const int keySize, const int valSize)
  {

    grow(keySize, finalKeySize, finalKeySpace, finalKeys);
    grow(valSize, finalValSize, finalValSpace, finalVals);
    memcpy(finalKeys + finalKeySize, keys, keySize);
    memcpy(finalVals + finalValSize, vals, valSize);
    finalKeySize += keySize;
    finalValSize += valSize;

  }//void

  PandaMessage::PandaMessage(const int pSingleKeySize, const int pSingleValSize, const bool pCopySendData)
  {
    singleKeySize = pSingleKeySize;
    singleValSize = pSingleValSize;
    copySendData  = pCopySendData;
  }

  PandaMessage::~PandaMessage()
  {
	  //TODO
  }

  oscpp::AsyncIORequest * PandaMessage::sendTo(const int rank,
                                                  void * const keys,
                                                  void * const vals,
                                                  const int keySize,
                                                  const int valSize)
  {
    PandaMessageIOData * data  = new PandaMessageIOData;
    data->flag = new volatile bool;
    data->waiting = new volatile bool;
    *data->flag = false;
    *data->waiting = false;
    if (copySendData)
    {
      if (keySize > 0 && keys != NULL)
      {
        data->keys = new char[keySize];
        memcpy(data->keys, keys, keySize);
      }
      else
      {
        data->keys = keys;
      }
      if (valSize > 0 && vals != NULL)
      {
        data->vals = new char[valSize];
        memcpy(data->vals, vals, valSize);
      }
      else
      {
        data->vals = vals;
      }
    }
    else
    {
      data->keys = keys;
      data->vals = vals;
    }
    data->keySize = keySize;
    data->valSize = valSize;
    data->rank = rank;

    if (rank == commRank)
    {
      data->counts = NULL;
    }
    else
    {
      data->counts = new int[2];
      data->counts[0] = keySize;
      data->counts[1] = valSize;
      data->done[0] = data->done[1] = data->done[2] = false;
    }
    PandaMessageIORequest * req = new PandaMessageIORequest(data->flag, data->waiting, data->keySize + data->valSize);
    data->cond = &req->condition();

    addDataLock.lock();
    needsToBeSent.push_back(data);
    addDataLock.unlock();
    return req;
  }
  oscpp::AsyncIORequest * PandaMessage::sendTo(const int rank,
                                                  void * const keys,
                                                  void * const vals,
                                                  int * const keySizes,
                                                  int * const valSizes,
                                                  const int numKeys,
                                                  const int numVals)
  {
    //TODO

  }
  void PandaMessage::init()
  {
    Message::init();
    zeroReqs.resize(commSize);
    zeroCount[0]  = zeroCount[1] = 0;
    finalKeySpace = 1048576;
    finalValSpace = 1048576;
    finalKeySize  = 0;
    finalValSize  = 0;
    finalKeys     = new char[finalKeySpace];
    finalVals     = new char[finalValSpace];
  }
  void PandaMessage::finalize()
  {
    delete [] finalKeys;
    delete [] finalVals;
  }
  void PandaMessage::run()
  {
    int finishedWorkers = 0;
    bool  * workerDone      = new bool[commSize];
    bool  * recvingCount    = new bool[commSize];
    int   * counts          = new int [commSize * 2];
    int  ** keyRecv         = new int*[commSize];
    int  ** valRecv         = new int*[commSize];
    MPI_Request * recvReqs  = new MPI_Request[commSize * 2];

    for (int i = 0; i < commSize; ++i)
    {
      workerDone[i] = false;
      recvingCount[i] = true;
      keyRecv[i] = NULL;
      valRecv[i] = NULL;
      MPI_Irecv(counts + i * 2, 2, MPI_INT, i, 0, MPI_COMM_WORLD, recvReqs + i * 2);
    }
    innerLoopDone = false;
    while (!innerLoopDone || finishedWorkers < commSize)
    {
      poll(finishedWorkers, workerDone, recvingCount, counts, keyRecv, valRecv, recvReqs);
      pollSends();
    }
    MPI_Waitall(commSize, &zeroReqs[0], MPI_STATUSES_IGNORE);

    delete [] workerDone;
    delete [] recvingCount;
    delete [] counts;
    delete [] keyRecv;
    delete [] valRecv;
    delete [] recvReqs;
  }
}
