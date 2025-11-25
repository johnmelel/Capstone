import React from 'react';
import './ConfirmModal.css';

const ConfirmModal = ({ isOpen, onConfirm, onCancel, message }) => {
    if (!isOpen) return null;

    return (
        <div className="confirm-modal-overlay" onClick={onCancel}>
            <div className="confirm-modal" onClick={(e) => e.stopPropagation()}>
                <div className="confirm-modal-content">
                    <p className="confirm-modal-message">{message}</p>
                    <div className="confirm-modal-buttons">
                        <button className="confirm-btn cancel" onClick={onCancel}>
                            Cancel
                        </button>
                        <button className="confirm-btn ok" onClick={onConfirm}>
                            OK
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ConfirmModal;
