//
//  detevthApp.swift
//  detevth
//
//  Created by jyhong on 19/01/2026.
//

import SwiftUI

@main
struct detevthApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
