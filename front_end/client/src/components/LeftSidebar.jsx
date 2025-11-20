import React, { useState, useEffect } from 'react';
import './LeftSidebar.css';

const LeftSidebar = ({ patients, conversations, selectedPatient, onPatientSelect, onNewChat, onLoadConversation }) => {
    const [showAllPatients, setShowAllPatients] = useState(true);

    // Debug: log conversations prop
    useEffect(() => {
        console.log('LeftSidebar received conversations:', conversations);
    }, [conversations]);

    return (
        <div className="left-sidebar">
            <button className="new-chat-btn" onClick={onNewChat}>
                <span className="plus-icon">+</span>
                New chat
            </button>

            <div className="patient-filter">
                <div className="filter-header">
                    <input
                        type="text"
                        placeholder="Patient Name..."
                        className="patient-search"
                    />
                    <button className="clear-btn" onClick={() => onPatientSelect(null)}>
                        Clear all
                    </button>
                </div>

                <div className="patient-list">
                    {patients && patients.length > 0 ? (
                        patients.map((patient) => (
                            <label key={patient.id} className="patient-item">
                                <input
                                    type="radio"
                                    name="patient"
                                    checked={selectedPatient === patient.id}
                                    onChange={() => onPatientSelect(patient.id)}
                                />
                                <span className="patient-name">{patient.name}</span>
                            </label>
                        ))
                    ) : (
                        <div className="empty-state-message">No patients loaded</div>
                    )}
                </div>
            </div>

            <div className="conversation-history">
                <div className="history-header">
                    <span>Your conversations</span>
                    <button className="clear-btn" onClick={() => { }}>
                        Clear all
                    </button>
                </div>

                <div className="conversation-list">
                    {conversations && conversations.length > 0 ? (
                        conversations.map((conv) => (
                            <div
                                key={conv.id}
                                className="conversation-item"
                                onClick={() => onLoadConversation(conv.id)}
                            >
                                <span className="conv-icon">💬</span>
                                <span className="conv-text">{conv.title}</span>
                            </div>
                        ))
                    ) : (
                        <div className="empty-state-message">No saved chats yet</div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default LeftSidebar;
