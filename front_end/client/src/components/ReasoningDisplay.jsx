import React, { useState } from 'react';
import './ReasoningDisplay.css';

const ReasoningDisplay = ({ steps, structured, unstructured }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    if (!steps && !structured && !unstructured) return null;

    const renderStructuredData = (data) => {
        if (Array.isArray(data)) {
            return (
                <ul>
                    {data.map((item, idx) => (
                        <li key={idx}>{item}</li>
                    ))}
                </ul>
            );
        }

        // Handle EMR worker response format
        if (data.sql_used) {
            const rowCount = data.row_count || 0;
            const rows = data.data?.rows || [];
            const displayRows = rows.slice(0, 5); // Show first 5 rows
            const hasMore = rows.length > 5;

            return (
                <div className="emr-data">
                    <div className="subsection">
                        <h5>SQL Query</h5>
                        <pre className="sql-query">{data.sql_used}</pre>
                    </div>

                    <div className="subsection">
                        <h5>Results Summary</h5>
                        <p>{data.summary || 'Query executed successfully'}</p>
                    </div>

                    {rows.length > 0 && (
                        <div className="subsection">
                            <h5>Data Retrieved</h5>
                            <p className="row-count">
                                Retrieved {rowCount} row{rowCount !== 1 ? 's' : ''} from database
                            </p>
                            <div className="data-preview">
                                <p className="preview-label">
                                    <strong>Sample data</strong> (showing {Math.min(5, rows.length)} of {rows.length}):
                                </p>
                                <pre className="data-rows">
                                    {JSON.stringify(displayRows, null, 2)}
                                </pre>
                                {hasMore && (
                                    <p className="more-rows">
                                        ... and {rows.length - 5} more rows
                                    </p>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            );
        }

        // Fallback for unknown format
        return <pre className="json-data">{JSON.stringify(data, null, 2)}</pre>;
    };

    return (
        <div className="reasoning-display">
            <button
                className="reasoning-toggle"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
                <span className="toggle-text">
                    Reasoning & Sources {steps ? `(${steps} steps)` : ''}
                </span>
            </button>

            {isExpanded && (
                <div className="reasoning-content">
                    {structured && (
                        <div className="source-section">
                            <h4>Structured Data (EMR)</h4>
                            {renderStructuredData(structured)}
                        </div>
                    )}

                    {unstructured && (
                        <div className="source-section">
                            <h4>Medical Knowledge</h4>
                            {Array.isArray(unstructured) ? (
                                <ul>
                                    {unstructured.map((item, idx) => (
                                        <li key={idx}>{item}</li>
                                    ))}
                                </ul>
                            ) : (
                                <pre className="json-data">{JSON.stringify(unstructured, null, 2)}</pre>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ReasoningDisplay;
