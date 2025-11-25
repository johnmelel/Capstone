import React, { useState, useEffect } from 'react';
import './LeftSidebar.css';

const LeftSidebar = ({ patients, conversations, selectedPatients, onPatientSelect, onNewChat, onLoadConversation, onClearConversations }) => {
    const [showAllPatients, setShowAllPatients] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');

    // Debug: log conversations prop
    useEffect(() => {
        console.log('LeftSidebar received conversations:', conversations);
    }, [conversations]);

    // Filter patients based on search term
    const filteredPatients = patients.filter(patient =>
        patient.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    const handleSearchKeyDown = (e) => {
        if (e.key === 'Enter' && filteredPatients.length > 0) {
            // Select the first filtered patient
            onPatientSelect(filteredPatients[0].id);
        }
    };

    const handleClearPatients = () => {
        // Clear all selected patients
        selectedPatients.forEach(patientId => {
            onPatientSelect(patientId); // Toggle each one off
        });
        setSearchTerm('');
    };

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
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        onKeyDown={handleSearchKeyDown}
                    />
                    <button className="clear-btn" onClick={handleClearPatients}>
                        Clear all
                    </button>
                </div>

                <div className="patient-list">
                    {filteredPatients && filteredPatients.length > 0 ? (
                        filteredPatients.map((patient) => (
                            <label key={patient.id} className="patient-item">
                                <input
                                    type="checkbox"
                                    name="patient"
                                    checked={selectedPatients.includes(patient.id)}
                                    onChange={() => onPatientSelect(patient.id)}
                                />
                                <span className="patient-name">{patient.name}</span>
                            </label>
                        ))
                    ) : (
                        <div className="empty-state-message">
                            {searchTerm ? 'No patients found' : 'No patients loaded'}
                        </div>
                    )}
                </div>
            </div>

            <div className="conversation-history">
                <div className="history-header">
                    <span>Your conversations</span>
                    <button className="clear-btn" onClick={onClearConversations}>
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
