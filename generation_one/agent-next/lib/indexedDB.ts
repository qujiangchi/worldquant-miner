export const openDatabase = async () => {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open('worldquant-miner', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      
      if (!db.objectStoreNames.contains('tools')) {
        const toolsStore = db.createObjectStore('tools', { keyPath: 'id' });
        toolsStore.createIndex('name', 'name', { unique: true });
      }
      
      if (!db.objectStoreNames.contains('contexts')) {
        const contextsStore = db.createObjectStore('contexts', { keyPath: 'id' });
        contextsStore.createIndex('name', 'name', { unique: true });
      }

      if (!db.objectStoreNames.contains('simulations')) {
        const simulationsStore = db.createObjectStore('simulations', { keyPath: 'id' });
        simulationsStore.createIndex('status', 'status', { unique: false });
        simulationsStore.createIndex('created_at', 'created_at', { unique: false });
      }
    };
  });
};

export const getAllFromStore = async (db: IDBDatabase, storeName: string) => {
  return new Promise<any[]>((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readonly');
    const store = transaction.objectStore(storeName);
    const request = store.getAll();

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

export const getFromStore = async (db: IDBDatabase, storeName: string, key: string) => {
  return new Promise<any>((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readonly');
    const store = transaction.objectStore(storeName);
    const request = store.get(key);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

export const putInStore = async (db: IDBDatabase, storeName: string, item: any) => {
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.put(item);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

export const deleteFromStore = async (db: IDBDatabase, storeName: string, key: string) => {
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.delete(key);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

// Simulation-specific functions
export const getSimulations = async (db: IDBDatabase, status?: string) => {
  return new Promise<any[]>((resolve, reject) => {
    const transaction = db.transaction('simulations', 'readonly');
    const store = transaction.objectStore('simulations');
    let request;

    if (status) {
      const index = store.index('status');
      request = index.getAll(status);
    } else {
      request = store.getAll();
    }

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

export const updateSimulation = async (db: IDBDatabase, simulation: any) => {
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction('simulations', 'readwrite');
    const store = transaction.objectStore('simulations');
    const request = store.put(simulation);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

export const addSimulation = async (db: IDBDatabase, simulation: any) => {
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction('simulations', 'readwrite');
    const store = transaction.objectStore('simulations');
    const request = store.add(simulation);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

export const deleteSimulation = async (db: IDBDatabase, id: string) => {
  return new Promise<void>((resolve, reject) => {
    const transaction = db.transaction('simulations', 'readwrite');
    const store = transaction.objectStore('simulations');
    const request = store.delete(id);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
}; 