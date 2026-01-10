'use client';

import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from '@headlessui/react';
import { useState } from 'react';
import { FaSpinner } from 'react-icons/fa';
import { apiClient } from '@/utils/api';

interface GenerateMasksModalProps {
  isOpen: boolean;
  onClose: () => void;
  datasetName: string;
  imagePath?: string; // Optional: for single image mask generation
  onComplete?: () => void;
}

export default function GenerateMasksModal({
  isOpen,
  onClose,
  datasetName,
  imagePath,
  onComplete,
}: GenerateMasksModalProps) {
  const [identifiers, setIdentifiers] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mergeTexts, setMergeTexts] = useState(true);

  const handleGenerate = async () => {
    if (!identifiers.trim()) {
      setError('Please enter at least one identifier');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const response = await apiClient.post('/api/masks/generate', {
        datasetName,
        imagePath: imagePath || null,
        identifiers: identifiers.split(',').map(id => id.trim()).filter(id => id),
        merge_texts: mergeTexts,
      });

      if (response.data.success) {
        onClose();
        if (onComplete) onComplete();
      } else {
        setError(response.data.error || 'Failed to generate masks');
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'An error occurred while generating masks');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleClose = () => {
    if (!isGenerating) {
      setIdentifiers('');
      setError(null);
      setMergeTexts(true);
      onClose();
    }
  };

  return (
    <Dialog open={isOpen} onClose={handleClose} className="relative z-50">
      <DialogBackdrop
        transition
        className="fixed inset-0 bg-gray-900/75 transition-opacity data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in"
      />

      <div className="fixed inset-0 z-10 w-screen overflow-y-auto">
        <div className="flex min-h-full items-center justify-center p-4">
          <DialogPanel
            transition
            className="relative transform overflow-hidden rounded-lg bg-gray-800 text-left shadow-xl transition-all data-closed:translate-y-4 data-closed:opacity-0 data-enter:duration-300 data-enter:ease-out data-leave:duration-200 data-leave:ease-in sm:my-8 sm:w-full sm:max-w-lg data-closed:sm:translate-y-0 data-closed:sm:scale-95"
          >
            <div className="bg-gray-800 px-6 pt-5 pb-4">
              <div className="text-center">
                <DialogTitle as="h3" className="text-lg font-semibold text-gray-200 mb-4">
                  Generate Mask{imagePath ? '' : 's'}
                </DialogTitle>

                <div className="text-left mb-4">
                  <p className="text-sm text-gray-400 mb-2">
                    {imagePath
                      ? 'Generate a mask for this image using SAM3 text prompts.'
                      : `Generate masks for all images in dataset: ${datasetName}`}
                  </p>
                  <p className="text-xs text-gray-500 mb-4">
                    Enter comma-separated text identifiers (e.g., "woman, person, character")
                  </p>

                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Identifiers<span className="text-red-500">*</span>
                  </label>
                  <input
                    type="text"
                    value={identifiers}
                    onChange={(e) => setIdentifiers(e.target.value)}
                    placeholder="woman, person, character"
                    disabled={isGenerating}
                    className="w-full rounded bg-gray-700 text-white p-2 border border-gray-600 focus:border-blue-500 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
                  />

                  <div className="mt-3 flex items-center gap-2">
                    <input
                      id="merge_texts"
                      type="checkbox"
                      checked={mergeTexts}
                      onChange={(e) => setMergeTexts(e.target.checked)}
                      disabled={isGenerating}
                      className="h-4 w-4 rounded bg-gray-700 border-gray-600 text-emerald-500 focus:ring-emerald-500 disabled:opacity-50"
                    />
                    <label htmlFor="merge_texts" className="text-sm text-gray-300">Merge identifiers into a single mask (default)</label>
                  </div>

                  {error && (
                    <p className="text-sm text-red-500 mt-2">{error}</p>
                  )}
                </div>

                {isGenerating && (
                  <div className="bg-blue-900/20 border border-blue-500/50 rounded p-4 mb-4">
                    <div className="flex items-center gap-2 text-blue-400 text-sm">
                      <FaSpinner className="animate-spin" />
                      <span>Generating masks... Please be patient, SAM3 is not fast.</span>
                    </div>
                    <p className="text-xs text-gray-400 mt-2">
                      This may take several minutes depending on the number of images. Do not close this window.
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="bg-gray-900 px-4 py-3 sm:flex sm:flex-row-reverse sm:px-6 gap-2">
              <button
                type="button"
                disabled={isGenerating}
                onClick={handleGenerate}
                className="inline-flex w-full justify-center rounded-md bg-emerald-600 px-3 py-2 text-sm font-semibold text-white shadow-sm hover:bg-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed sm:w-auto"
              >
                {isGenerating ? 'Generating...' : 'Generate'}
              </button>
              <button
                type="button"
                disabled={isGenerating}
                onClick={handleClose}
                className="mt-3 inline-flex w-full justify-center rounded-md bg-gray-700 px-3 py-2 text-sm font-semibold text-gray-200 shadow-sm hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed sm:mt-0 sm:w-auto"
              >
                Cancel
              </button>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}
