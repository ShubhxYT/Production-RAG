import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { fetchDocuments } from '../lib/api';
import DocumentCard from '../components/library/DocumentCard';
import HealthMetrics from '../components/library/HealthMetrics';
import UploadModal from '../components/library/UploadModal';

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.3 } },
};

export default function Library() {
  const queryClient = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ['documents'],
    queryFn: fetchDocuments,
    retry: false,
  });
  const [showUpload, setShowUpload] = useState(false);

  const handleUploadSuccess = () => {
    queryClient.invalidateQueries({ queryKey: ['documents'] });
  };

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="flex-1 p-6 lg:p-10"
    >
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <header className="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12">
          <div>
            <h1 className="text-3xl font-extrabold font-headline text-on-surface tracking-tight mb-2">
              Knowledge Base
            </h1>
            <p className="text-on-surface-variant font-medium">
              Manage and monitor your indexed intellectual assets.
            </p>
          </div>
          <button
            onClick={() => setShowUpload(true)}
            className="flex items-center justify-center gap-2 px-8 py-4 primary-gradient text-white rounded-full font-headline font-bold text-sm tracking-widest shadow-xl hover:scale-[0.98] transition-transform w-full md:w-auto"
          >
            <span className="material-symbols-outlined">upload_file</span>
            UPLOAD DOCUMENT
          </button>
        </header>

        <UploadModal
          open={showUpload}
          onClose={() => setShowUpload(false)}
          onSuccess={handleUploadSuccess}
        />

        {/* Document Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
          {isLoading
            ? Array.from({ length: 6 }).map((_, i) => (
                <div
                  key={i}
                  className="bg-slate-800/50 p-6 rounded-xl animate-pulse"
                >
                  <div className="h-10 w-10 bg-slate-700 rounded-lg mb-4" />
                  <div className="h-5 bg-slate-700 rounded w-3/4 mb-2" />
                  <div className="h-3 bg-slate-700 rounded w-1/2 mb-6" />
                  <div className="h-3 bg-slate-700 rounded w-full" />
                </div>
              ))
            : data?.documents.map((doc) => (
                <DocumentCard key={doc.id} document={doc} />
              ))}
          {!isLoading && data && data.documents.length === 0 && (
            <div className="col-span-full text-center py-16 text-slate-500">
              <span className="material-symbols-outlined text-4xl mb-4 block">
                folder_off
              </span>
              <p className="font-headline font-bold text-lg mb-2">
                No documents yet
              </p>
              <p className="text-sm">
                Upload your first document to get started.
              </p>
            </div>
          )}
        </div>

        {/* Knowledge Base Health */}
        <HealthMetrics />

        <footer className="mt-16 text-center text-outline text-xs font-medium tracking-widest uppercase">
          © 2024 FullRag Systems • Intellectual Integrity Secured
        </footer>
      </div>
    </motion.div>
  );
}
