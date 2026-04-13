import { useCallback, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { uploadDocument, getUploadStatus } from '../../lib/api';
import type { UploadStage } from '../../lib/types';

const ACCEPTED_MIME: Record<string, string> = {
  '.pdf': 'application/pdf',
  '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  '.html': 'text/html',
  '.htm': 'text/html',
  '.md': 'text/markdown',
};
const ACCEPTED_EXTS = Object.keys(ACCEPTED_MIME).join(', ');
const ACCEPTED_ACCEPT = Object.values(ACCEPTED_MIME).join(',');
const MAX_SIZE_MB = 50;

const STAGE_LABELS: Record<UploadStage, string> = {
  queued: 'Queued…',
  loading: 'Parsing document…',
  saving: 'Saving to knowledge base…',
  embedding: 'Generating embeddings…',
  indexing: 'Indexing vectors…',
  complete: 'Complete',
  error: 'Failed',
};

interface Props {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

type ModalState =
  | { kind: 'idle'; error?: string }
  | { kind: 'uploading'; fileName: string; xhrProgress: number }
  | { kind: 'processing'; fileName: string; stage: UploadStage; progress: number }
  | { kind: 'complete'; fileName: string }
  | { kind: 'error'; message: string };

export default function UploadModal({ open, onClose, onSuccess }: Props) {
  const [state, setState] = useState<ModalState>({ kind: 'idle' });
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => {
    if (pollingRef.current !== null) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  };

  const handleClose = () => {
    if (state.kind === 'uploading' || state.kind === 'processing') return; // block close while running
    stopPolling();
    setState({ kind: 'idle' });
    onClose();
  };

  const validateFile = (file: File): string | null => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!Object.keys(ACCEPTED_MIME).includes(ext)) {
      return `Unsupported type "${ext}". Accepted: ${ACCEPTED_EXTS}`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File exceeds ${MAX_SIZE_MB} MB limit.`;
    }
    return null;
  };

  const startUpload = useCallback(async (file: File) => {
    const err = validateFile(file);
    if (err) {
      setState({ kind: 'idle', error: err });
      return;
    }

    setState({ kind: 'uploading', fileName: file.name, xhrProgress: 0 });

    try {
      const { job_id } = await uploadDocument(file, (pct) => {
        setState((prev) =>
          prev.kind === 'uploading' ? { ...prev, xhrProgress: pct } : prev,
        );
      });

      setState({ kind: 'processing', fileName: file.name, stage: 'queued', progress: 20 });

      // Poll status every 500 ms
      pollingRef.current = setInterval(async () => {
        try {
          const status = await getUploadStatus(job_id);

          if (status.stage === 'complete') {
            stopPolling();
            setState({ kind: 'complete', fileName: file.name });
            onSuccess();
          } else if (status.stage === 'error') {
            stopPolling();
            setState({ kind: 'error', message: status.error ?? 'Unknown error' });
          } else {
            setState({
              kind: 'processing',
              fileName: file.name,
              stage: status.stage,
              progress: Math.max(status.progress, 20),
            });
          }
        } catch {
          stopPolling();
          setState({ kind: 'error', message: 'Lost connection to server.' });
        }
      }, 500);
    } catch (e: unknown) {
      const msg =
        e instanceof Error
          ? e.message
          : (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
            'Upload failed.';
      setState({ kind: 'error', message: msg });
    }
  }, [onSuccess]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) startUpload(file);
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) startUpload(file);
  };

  const progressPct =
    state.kind === 'uploading'
      ? state.xhrProgress
      : state.kind === 'processing'
      ? state.progress
      : state.kind === 'complete'
      ? 100
      : 0;

  const stageLabel =
    state.kind === 'uploading'
      ? 'Uploading file…'
      : state.kind === 'processing'
      ? STAGE_LABELS[state.stage]
      : state.kind === 'complete'
      ? 'Complete'
      : '';

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          key="backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
          onClick={handleClose}
        >
          <motion.div
            key="panel"
            initial={{ opacity: 0, scale: 0.95, y: 16 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 16 }}
            transition={{ duration: 0.2 }}
            className="bg-slate-900 border border-white/10 rounded-2xl p-8 w-full max-w-md mx-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between mb-6">
              <h3 className="font-headline font-bold text-xl text-white">Upload Document</h3>
              {state.kind !== 'uploading' && state.kind !== 'processing' && (
                <button
                  onClick={handleClose}
                  className="text-slate-500 hover:text-white transition-colors"
                  aria-label="Close"
                >
                  <span className="material-symbols-outlined">close</span>
                </button>
              )}
            </div>

            {/* Idle / Drop Zone */}
            {(state.kind === 'idle') && (
              <>
                <div
                  onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`border-2 border-dashed rounded-xl p-10 flex flex-col items-center justify-center gap-3 cursor-pointer transition-colors select-none
                    ${dragging ? 'border-indigo-400 bg-indigo-950/30' : 'border-slate-700 hover:border-indigo-500 hover:bg-slate-800/40'}`}
                >
                  <span className="material-symbols-outlined text-4xl text-indigo-400">upload_file</span>
                  <p className="font-headline font-bold text-white text-sm">
                    Drag &amp; drop or <span className="text-indigo-400">browse</span>
                  </p>
                  <p className="text-xs text-slate-500">{ACCEPTED_EXTS} · max {MAX_SIZE_MB} MB</p>
                </div>
                {state.error && (
                  <p className="mt-4 text-red-400 text-sm text-center">{state.error}</p>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept={ACCEPTED_ACCEPT}
                  className="hidden"
                  onChange={handleFileChange}
                />
              </>
            )}

            {/* Uploading / Processing */}
            {(state.kind === 'uploading' || state.kind === 'processing') && (
              <div className="space-y-5">
                <div className="flex items-center gap-3 bg-slate-800 rounded-xl px-4 py-3">
                  <span className="material-symbols-outlined text-indigo-400">description</span>
                  <span className="text-sm font-medium text-white truncate">
                    {state.kind === 'uploading' ? state.fileName : state.fileName}
                  </span>
                </div>

                {/* Progress bar */}
                <div>
                  <div className="flex justify-between text-xs text-slate-400 mb-2">
                    <span>{stageLabel}</span>
                    <span>{progressPct}%</span>
                  </div>
                  <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                    <motion.div
                      className="h-full primary-gradient rounded-full"
                      initial={{ width: '0%' }}
                      animate={{ width: `${progressPct}%` }}
                      transition={{ ease: 'easeOut', duration: 0.4 }}
                    />
                  </div>
                </div>

                {/* Stage pipeline dots */}
                <div className="flex items-center gap-1.5 flex-wrap">
                  {(['loading', 'saving', 'embedding', 'indexing', 'complete'] as UploadStage[]).map(
                    (s) => {
                      const done =
                        progressPct >=
                        { loading: 20, saving: 60, embedding: 80, indexing: 95, complete: 100 }[s]!;
                      const active =
                        state.kind === 'processing' && state.stage === s;
                      return (
                        <span
                          key={s}
                          className={`text-[10px] font-bold px-2 py-0.5 rounded-full tracking-wide uppercase transition-colors
                            ${done ? 'bg-indigo-700/60 text-indigo-200' : active ? 'bg-indigo-500/30 text-indigo-300 animate-pulse' : 'bg-slate-800 text-slate-600'}`}
                        >
                          {STAGE_LABELS[s].replace('…', '')}
                        </span>
                      );
                    },
                  )}
                </div>
              </div>
            )}

            {/* Complete */}
            {state.kind === 'complete' && (
              <div className="flex flex-col items-center gap-5 py-2">
                <div className="w-14 h-14 rounded-full bg-emerald-900/40 flex items-center justify-center">
                  <span className="material-symbols-outlined text-3xl text-emerald-400">check_circle</span>
                </div>
                <div className="text-center">
                  <p className="font-headline font-bold text-white text-lg mb-1">Document Indexed</p>
                  <p className="text-slate-400 text-sm truncate max-w-xs">{state.fileName}</p>
                </div>
                <button
                  onClick={handleClose}
                  className="w-full py-3 primary-gradient text-white rounded-xl font-headline font-bold text-sm"
                >
                  Done
                </button>
              </div>
            )}

            {/* Error */}
            {state.kind === 'error' && (
              <div className="flex flex-col items-center gap-5 py-2">
                <div className="w-14 h-14 rounded-full bg-red-900/40 flex items-center justify-center">
                  <span className="material-symbols-outlined text-3xl text-red-400">error</span>
                </div>
                <div className="text-center">
                  <p className="font-headline font-bold text-white text-lg mb-1">Upload Failed</p>
                  <p className="text-slate-400 text-sm">{state.message}</p>
                </div>
                <button
                  onClick={() => setState({ kind: 'idle' })}
                  className="w-full py-3 primary-gradient text-white rounded-xl font-headline font-bold text-sm"
                >
                  Try Again
                </button>
              </div>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
